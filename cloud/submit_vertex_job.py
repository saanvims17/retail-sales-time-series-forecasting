from google.cloud import aiplatform

PROJECT_ID = "linen-compiler-510211-c2"
REGION = "us-central1"
BUCKET = "gs://retail-forecasting-saanvi-2026"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET
)

job = aiplatform.CustomJob.from_local_script(
    display_name="retail-xgboost-training",
    script_path="vertex_xgboost.py",
    container_uri=(
        "us-docker.pkg.dev/vertex-ai/"
        "training/xgboost-cpu.2-1:latest"
    ),
    args=[
        "--train-data",
        f"{BUCKET}/data/train_features.csv",
        "--validation-data",
        f"{BUCKET}/data/validation_features.csv",
        "--model-dir",
        "/gcs/retail-forecasting-saanvi-2026/models/retail-xgboost"
    ],
    replica_count=1,
    machine_type="n1-standard-4"
)

job.run(sync=True)

print("Vertex AI training job completed!")