import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from huggingface_hub import login
import os

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

dataset = load_dataset("jimmyabhinav28/tourism_Dataset", split="train")
df = dataset.to_pandas()

df_cleaned = df.drop(columns=['CustomerID'], errors='ignore')
numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
for col in numeric_cols:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)

train_ds = Dataset.from_pandas(train_df, preserve_index=False)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)
cleaned_dict = DatasetDict({"train": train_ds, "test": test_ds})
cleaned_dict.push_to_hub("jimmyabhinav28/tourism_Dataset_Cleaned")
print("Data prepped and pushed!")
