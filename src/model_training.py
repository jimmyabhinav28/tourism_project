import pandas as pd
import joblib
from datasets import load_dataset
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo, login
import os

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

dataset = load_dataset("jimmyabhinav28/tourism_Dataset_Cleaned")
train_df = dataset['train'].to_pandas()

X_train = train_df.drop(columns=['ProdTaken'])
y_train = train_df['ProdTaken']
X_train_num = X_train.select_dtypes(include=['number'])

model = XGBClassifier(subsample=1.0, n_estimators=100, max_depth=7, learning_rate=0.05)
model.fit(X_train_num, y_train)

os.makedirs("model_building", exist_ok=True)
joblib.dump(model, "model_building/best_xgboost_model.joblib")

api = HfApi()
repo_id = "jimmyabhinav28/Wellness-Tourism-XGBoost-Model"
create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="model_building/best_xgboost_model.joblib", path_in_repo="best_xgboost_model.joblib", repo_id=repo_id, repo_type="model")
print("Model trained and pushed!")
