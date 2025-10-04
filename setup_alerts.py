#!/usr/bin/env python3
"""
This script sets up automated alerts for SageMaker Model Monitor.

It creates an SNS topic and an EventBridge rule that listens for
Model Monitor jobs that complete with violations (indicating drift)
and sends a notification to the specified email address.
"""
import boto3
import json

def main():
    """
    Creates the necessary AWS resources for monitoring alerts.
    """
    # --- Configuration ---
    # IMPORTANT: Replace with your email address to receive alerts.
    email_address = "your-email@example.com"
    
    # This should match the schedule name from setup_model_monitor.py
    monitor_schedule_name = "pando2-co2-monitor-schedule" 
    
    # Use your AWS region
    aws_region = boto3.Session().region_name
    aws_account_id = boto3.client("sts").get_caller_identity()["Account"]
    # -------------------

    sns_client = boto3.client("sns")
    events_client = boto3.client("events")

    print("Setting up alerts...")

    # 1. Create an SNS Topic
    topic_name = "SageMakerModelMonitorAlerts"
    topic_arn = sns_client.create_topic(Name=topic_name)["TopicArn"]
    print(f"✅ SNS Topic created: {topic_arn}")

    # 2. Subscribe your email to the SNS Topic
    sns_client.subscribe(
        TopicArn=topic_arn,
        Protocol="email",
        Endpoint=email_address
    )
    print(f"✅ Subscription pending for {email_address}. Please check your email to confirm.")

    # 3. Create an EventBridge Rule
    rule_name = "SageMakerMonitorViolationRule"
    
    # This event pattern specifically targets your monitoring schedule
    # when its status is "CompletedWithViolations".
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
        Description="Triggers when a SageMaker monitoring job detects violations."
    )
    print(f"✅ EventBridge rule '{rule_name}' created.")

    # 4. Set the SNS Topic as the target for the EventBridge Rule
    events_client.put_targets(
        Rule=rule_name,
        Targets=[{
            "Id": "SendToSNSTopic",
            "Arn": topic_arn
        }]
    )
    print("✅ SNS topic set as the target for the rule.")
    print("\n🎉 Alerting setup is complete! You will receive an email when data drift is detected.")

if __name__ == "__main__":
    main()