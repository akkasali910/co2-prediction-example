"""
This script defines and launches the complete SageMaker MLOps Pipeline for
the Pando2 CO2 regression model.

The pipeline includes:
- Data Preparation (Processing Step)
- Hyperparameter Tuning (Tuning Step)
- Retraining the best model (Training Step)
- Model Evaluation (Processing Step)
- Conditional Model Registration (Condition Step & RegisterModel)
- Conditional Model Deployment (Lambda Step)
"""
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig,
)
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TuningStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.workflow.check_steps import (
    ClarifyCheckStep,
    DataQualityCheckConfig,
    CheckJobConfig,
    ModelQualityCheckConfig,
    ModelBiasCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

def get_pipeline(role, pipeline_name, default_bucket):
    """Creates and returns a SageMaker Pipeline instance."""
    sagemaker_session = sagemaker.Session(default_bucket=default_bucket)
    
    # --- Parameters ---
    processing_instance_type = "ml.m5.xlarge"
    training_instance_type = "ml.m5.xlarge"
    
    # --- Step 1: Data Preparation ---
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1", role=role, instance_type=processing_instance_type, instance_count=1,
        base_job_name="co2-data-prep"
    )
    step_prepare_data = ProcessingStep(
        name="PrepareCO2Data",
        processor=sklearn_processor,
        code="scripts/preprocess.py",
        inputs=[sagemaker.inputs.ProcessingInput(source=f"s3://{default_bucket}/input/", destination="/opt/ml/processing/input")],
        outputs=[
            sagemaker.outputs.ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            sagemaker.outputs.ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
            sagemaker.outputs.ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test"),
            sagemaker.outputs.ProcessingOutput(output_name="feature_count", source="/opt/ml/processing/output/feature_count"),
        ]
    )

    # --- NEW: Data Quality Baseline Step ---
    # This step creates a baseline of statistics and constraints from the training data.
    # This baseline is used later by Model Monitor to detect data drift in production.
    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
    )
    data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=step_prepare_data.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        dataset_format=sagemaker.model_monitor.dataset_format.DatasetFormat.csv(header=False),
        output_s3_uri=f"s3://{default_bucket}/monitoring/data-quality"
    )
    step_data_quality_check = QualityCheckStep(
        name="CheckDataQuality",
        check_job_config=check_job_config,
        quality_check_config=data_quality_check_config,
        skip_check=True, # We only want to generate the baseline, not fail the pipeline
    )

    # --- Step 2: Hyperparameter Tuning ---
    xgb_estimator = XGBoost(
        entry_point="scripts/train.py",
        framework_version="1.7-1",
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=f"s3://{default_bucket}/co2-output/training",
    )
    
    hyperparameter_ranges = {
        "n_estimators": IntegerParameter(100, 500),
        "max_depth": IntegerParameter(3, 10),
        "learning_rate": ContinuousParameter(0.01, 0.3),
        "subsample": ContinuousParameter(0.6, 1.0),
    }

    tuner = HyperparameterTuner(
        estimator=xgb_estimator,
        objective_metric_name="validation:rmse",
        objective_type="Minimize",
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=20,
        max_parallel_jobs=5,
        strategy="Bayesian",
    )

    step_tune_model = TuningStep(
        name="TuneCO2Model",
        tuner=tuner,
        inputs={
            "train": sagemaker.inputs.TrainingInput(s3_data=step_prepare_data.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri),
            "validation": sagemaker.inputs.TrainingInput(s3_data=step_prepare_data.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri),
        }
    )

    # --- Step 3: Train Best Model on Full Dataset ---
    # This step takes the best hyperparameters from the tuning job and retrains the
    # model on the combined training and validation datasets for maximum robustness.
    step_train_best_model = TrainingStep(
        name="TrainBestCO2Model",
        estimator=xgb_estimator,
        # Use the best hyperparameters found by the tuning step
        hyperparameters=step_tune_model.best_hyperparameters,
        inputs={
            # The train.py script is designed to concatenate these two channels
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_prepare_data.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            ),
            "validation": sagemaker.inputs.TrainingInput(
                s3_data=step_prepare_data.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri
            ),
        },
    )

    # --- NEW: Bias and Explainability Step (SageMaker Clarify) ---
    # This step runs after training to analyze the model for bias and explain its predictions.
    clarify_processor = sagemaker.clarify.SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=sagemaker_session,
    )
    
    # Define a PropertyFile to read the feature count from the preprocessing step's output
    feature_count_report = PropertyFile(
        name="FeatureCountReport",
        output_name="feature_count",
        path="feature_count.json"
    )
    step_prepare_data.properties.ProcessingOutputConfig.Outputs["feature_count"]._set_prop_file(feature_count_report)
    num_features = feature_count_report.properties["num_features"]

    # Configure Clarify to analyze the data, specifying the target label and features.
    # For bias, we'll use 'occupancy' as a facet to see if the model behaves differently
    # for different occupancy levels.
    clarify_data_config = DataConfig(
        s3_data_input_path=step_prepare_data.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=f"s3://{default_bucket}/clarify-output",
        label="CO2_PPM", # The target variable, matching preprocess.py
        headers=[f"feature_{i}" for i in range(num_features)] + ["CO2_PPM"], # Use dynamic feature count
        dataset_type="text/csv",
    )

    # Configure the model for Clarify to use for analysis
    clarify_model_config = ModelConfig(
        model_name=step_train_best_model.properties.ModelName,
        instance_type=training_instance_type,
        instance_count=1,
        accept_type="text/csv",
    )

    # Configure the bias check (e.g., check for bias related to 'occupancy')
    # This requires knowing which column index corresponds to 'occupancy'. We'll assume it's 1.
    bias_config = BiasConfig(
        label_values_or_threshold=[1], # Not applicable for regression, but required
        facet_name="feature_1", # Assuming 'occupancy' is the second feature (index 1)
    )

    # Configure SHAP for explainability
    shap_config = SHAPConfig(baseline=[[0]*num_features]) # Use dynamic feature count

    step_clarify_check = ClarifyCheckStep(
        name="CheckModelBiasAndExplainability",
        clarify_job_config=clarify_processor.run_config(
            model_config=clarify_model_config, 
            data_config=clarify_data_config, 
            analysis_config=bias_config, 
            shap_config=shap_config
        ),
        # Set skip_check to False to enforce the quality gate
        skip_check=False,
        # Define the thresholds for bias metrics.
        # The pipeline will fail if the absolute value of DPL exceeds 0.2.
        model_bias_check_config=ModelBiasCheckConfig(
            data_config=clarify_data_config,
            bias_config=BiasConfig(label_values_or_threshold=[1], facet_name="feature_1", report_computed_metrics=True, threshold=0.2)
        )
    )

    # --- NEW: Bias Mitigation Step (Clarify Pre-training Bias) ---
    # This step runs a Clarify processing job to generate instance weights
    # that can be used to mitigate pre-training bias.
    step_mitigate_bias = ProcessingStep(
        name="MitigateDataBias",
        processor=clarify_processor,
        inputs=[
            ProcessingInput(
                source=step_prepare_data.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                output_name="debiased_data",
            )
        ],
        # The job_arguments tell Clarify to compute weights
        job_arguments=[
            f"--data-config {clarify_data_config.to_json()}",
            f"--analysis-config {bias_config.to_json()}",
            "--methods 'pre_training_bias'",
            "--debiasing-methods 'instance_weighting'",
        ],
    )

    # --- Step 4: Model Evaluation ---
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    
    step_evaluate_model = ProcessingStep(
        name="EvaluateCO2Model",
        processor=sklearn_processor,
        # The evaluation script now takes the retrained model as input
        # instead of the best model from the tuning job.
        code="scripts/evaluate.py",
        inputs=[
            sagemaker.inputs.ProcessingInput(source=step_train_best_model.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            sagemaker.inputs.ProcessingInput(source=step_prepare_data.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test")
        ],
        outputs=[sagemaker.outputs.ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        property_files=[evaluation_report]
    )

    # --- Step 5: Conditional Model Registration ---
    model_package_group_name = "Pando2-CO2-Regression-Models"
    
    step_register_model = RegisterModel(
        name="RegisterCO2Model",
        estimator=xgb_estimator,
        # The model being registered is now the one trained on the full dataset.
        model_data=step_train_best_model.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        model_package_group_name=model_package_group_name,
        model_metrics=sagemaker.model_metrics.ModelMetrics(
            model_statistics=sagemaker.model_metrics.MetricsSource(
                s3_uri=evaluation_report.properties.S3Url,
                content_type="application/json"
            )
        )
    )

    cond_register = ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_evaluate_model.properties.EvaluationReport.path, "regression_metrics.r2_score.value"),
        right=0.75  # Register model only if R-squared is >= 0.75
    )

    # --- Step 6: Conditional Model Deployment ---
    # This step uses a Lambda function to deploy the registered model package
    # to a SageMaker endpoint. It only runs if the registration step succeeds.
    # NOTE: You must create a Lambda function with the code from 'scripts/deploy.py'
    # and an IAM role that allows it to perform sagemaker:CreateModel,
    # sagemaker:CreateEndpointConfig, and sagemaker:CreateEndpoint/UpdateEndpoint.
    step_deploy_model = LambdaStep(
        name="DeployCO2Model",
        lambda_func_arn="arn:aws:lambda:YOUR_REGION:YOUR_ACCOUNT_ID:function:sagemaker-deploy-function", # <-- IMPORTANT: UPDATE THIS ARN
        inputs={
            "model_package_arn": step_register_model.properties.ModelPackageArn,
            "endpoint_name": "co2-prediction-endpoint",
            "role_arn": role,
            "instance_type": "ml.m5.large",
            "s3_bucket": default_bucket, # Pass the bucket name
            "instance_count": 1,
        },
    )

    # --- Step 7: Cleanup Step (Else Branch) ---
    # This step runs if the model performance is NOT good enough.
    # It cleans up the old endpoint to prevent it from running indefinitely.
    # NOTE: You must create a Lambda function with the code from 'scripts/cleanup.py'
    # and an IAM role that allows sagemaker:DeleteEndpoint.
    step_cleanup_endpoint = LambdaStep(
        name="CleanupOldEndpoint",
        lambda_func_arn="arn:aws:lambda:YOUR_REGION:YOUR_ACCOUNT_ID:function:sagemaker-cleanup-function", # <-- IMPORTANT: UPDATE THIS ARN
        inputs={
            "endpoint_name": "co2-prediction-endpoint",
        },
    )

    step_conditional_register = ConditionStep(
        name="CheckR2ScoreAndRegister",
        conditions=[cond_register],
        # If the condition is met, register the model and then deploy it.
        if_steps=[step_register_model, step_deploy_model],
        # If the condition is NOT met, run the cleanup step to delete the old endpoint.
        # This is a good practice to ensure you don't pay for an endpoint that
        # is now outdated by a new model (even if the new model wasn't good enough
        # to be deployed itself).
        else_steps=[step_cleanup_endpoint]
    )

    # --- Construct and return the Pipeline ---
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[
            step_prepare_data,
            step_data_quality_check, # New step for data quality baseline
            step_tune_model,
            step_train_best_model,
            step_mitigate_bias, # New step to generate de-biasing weights
            step_clarify_check, # New step for bias and explainability
            step_evaluate_model,
            step_conditional_register,
        ]
    )
    return pipeline

if __name__ == "__main__":
    role = sagemaker.get_execution_role()
    default_bucket = sagemaker.Session().default_bucket()
    pipeline_name = "Pando2-CO2-Regression-Pipeline"

    pipeline = get_pipeline(role, pipeline_name, default_bucket)
    
    print(f"Upserting pipeline: {pipeline_name}...")
    pipeline.upsert(role_arn=role)
    print("✅ Pipeline upserted successfully.")
    
    print("Starting pipeline execution...")
    execution = pipeline.start()
    print(f"✅ Pipeline execution started with ARN: {execution.arn}")
