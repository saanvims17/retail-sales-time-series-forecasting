from google.cloud import aiplatform

PROJECT_ID = "linen-compiler-510211-c2"
REGION = "us-central1"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION
)

model = aiplatform.Model.upload(
    display_name="retail-xgboost-bst-model",
    artifact_uri=(
        "gs://retail-forecasting-saanvi-2026/"
        "models/local-xgboost-bst/"
    ),
    serving_container_image_uri=(
        "us-docker.pkg.dev/vertex-ai/"
        "prediction/xgboost-cpu.2-1:latest"
    )
)

print("BST model uploaded successfully!")
print("Model:", model.resource_name)