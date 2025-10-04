#!/usr/bin/env python3
"""
This script sets up an automated retraining trigger for SageMaker Model Monitor.

It creates:
1. An SNS topic for email notifications.
2. An EventBridge rule that listens for Model Monitor jobs with violations.
3. Two targets for the rule:
    - The SNS topic to send an alert.
    - A Lambda function that starts a new SageMaker Pipeline execution.
"""
import boto3
import json

def main():
    """
    Creates the necessary AWS resources for automated retraining.
    """
    # --- Configuration ---
    # IMPORTANT: Replace with your details.
    email_address = "your-email@example.com"
    
    # This Lambda function should be created to start a pipeline execution.
    # It's the same one used for scheduling.
    pipeline_starter_lambda_arn = "arn:aws:lambda:YOUR_REGION:YOUR_ACCOUNT_ID:function:sagemaker-pipeline-starter"
    
    # This should match the schedule name from setup_model_monitor.py
    monitor_schedule_name = "pando2-co2-monitor-schedule"
    # -------------------

    sns_client = boto3.client("sns")
    events_client = boto3.client("events")

    print("Setting up automated retraining trigger...")

    # 1. Create an SNS Topic for notifications
    topic_name = "SageMakerModelMonitorAlerts"
    try:
        topic_arn = sns_client.create_topic(Name=topic_name)["TopicArn"]
        print(f"✅ SNS Topic created: {topic_arn}")

        # 2. Subscribe your email to the SNS Topic
        sns_client.subscribe(
            TopicArn=topic_arn,
            Protocol="email",
            Endpoint=email_address
        )
        print(f"✅ Subscription pending for {email_address}. Please check your email to confirm.")
    except Exception as e:
        print(f"⚠️  Could not create SNS topic (it might already exist). Error: {e}")
        # If it already exists, construct the ARN to continue
        region = boto3.Session().region_name
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        topic_arn = f"arn:aws:sns:{region}:{account_id}:{topic_name}"

    # 3. Create an EventBridge Rule
    rule_name = "SageMakerMonitorViolationRetrainRule"
    
    event_pattern = {
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Model Monitoring Job Schedule Status Change"],
        "detail": {
            "monitoringScheduleName": [monitor_schedule_name],
            "monitoringJobStatus": ["CompletedWithViolations"]
        }
    }

    events_client.put_rule(
        Name=rule_name,
        EventPattern=json.dumps(event_pattern),
        State="ENABLED",
        Description="Triggers retraining pipeline when a SageMaker monitoring job detects violations."
    )
    print(f"✅ EventBridge rule '{rule_name}' created.")

    # 4. Set the targets for the EventBridge Rule (SNS and Lambda)
    events_client.put_targets(
        Rule=rule_name,
        Targets=[
            {
                "Id": "SendAlertToSNSTopic",
                "Arn": topic_arn
            },
            {
                "Id": "TriggerRetrainingPipeline",
                "Arn": pipeline_starter_lambda_arn
            }
        ]
    )
    print("✅ Set SNS Topic and Lambda function as targets for the rule.")
    print("\n🎉 Automated retraining setup is complete!")

if __name__ == "__main__":
    main()