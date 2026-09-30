import joblib

model = joblib.load("vertex_model/model.pkl")

model.get_booster().save_model("vertex_model/model.bst")

print("Model converted successfully!")
print("Features:", model.get_booster().num_features())