import argparse, boto3, time, json, os, sys

def create_or_update_endpoint(sm, endpoint_name, image_uri, role_arn):
    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    # Create or update Model
    try:
        sm.describe_model(ModelName=model_name)
        print("Updating model...")
        sm.delete_model(ModelName=model_name)
        waiter = sm.get_waiter('model_deleted')
        # Note: model_deleted waiter doesn't exist; quick sleep instead
        time.sleep(2)
    except sm.exceptions.ClientError:
        pass

    print("Creating model with image:", image_uri)
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={"Image": image_uri}
    )

    # Create or update EndpointConfig
    try:
        sm.describe_endpoint_config(EndpointConfigName=config_name)
        print("Deleting old endpoint config...")
        sm.delete_endpoint_config(EndpointConfigName=config_name)
        time.sleep(2)
    except sm.exceptions.ClientError:
        pass

    print("Creating endpoint config...")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": "ml.m5.large",
            "InitialVariantWeight": 1.0
        }]
    )

    # Create or update Endpoint
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        print("Endpoint exists with status:", status)
        print("Updating endpoint...")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    except sm.exceptions.ClientError:
        print("Creating endpoint...")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

    print("Waiting for InService...")
    waiter = sm.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    print("Endpoint is InService.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-uri", required=True)
    ap.add_argument("--role-arn", required=True)
    ap.add_argument("--endpoint", required=True)
    args = ap.parse_args()

    sm = boto3.client("sagemaker")
    create_or_update_endpoint(sm, args.endpoint, args.image_uri, args.role_arn)
