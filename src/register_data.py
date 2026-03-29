import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import os

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

# Pointing to the data folder shown in your screenshot
csv_path = "data/tourism.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub("jimmyabhinav28/tourism_Dataset")
    print("Data registered successfully!")
else:
    print(f"Error: {csv_path} not found.")
    exit(1)
