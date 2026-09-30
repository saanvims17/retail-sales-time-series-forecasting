from google.cloud import aiplatform

PROJECT_ID = "linen-compiler-510211-c2"
REGION = "us-central1"
MODEL_ID = "2564422305543356416"

aiplatform.init(
    project=PROJECT_ID,
    location=REGION
)

model = aiplatform.Model(
    model_name=MODEL_ID
)

endpoint = aiplatform.Endpoint.create(
    display_name="retail-xgboost-bst-endpoint"
)

endpoint.deploy(
    model=model,
    deployed_model_display_name="retail-xgboost-bst",
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=1
)

print("BST model deployed successfully!")
print("Endpoint:", endpoint.resource_name)