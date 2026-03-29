from huggingface_hub import HfApi, create_repo
import os

# Configuration
hf_username = "jimmyabhinav28"
space_name = "Wellness-Tourism-Predictor"
repo_id = f"{hf_username}/{space_name}"

api = HfApi()

print(f"Creating Hugging Face Space: {repo_id}...")
# Create a Docker space
create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

print("Uploading deployment files...")
# Push the entire deployment folder to the Space
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space"
)

print(f"Deployment successfully pushed! View your app at: https://huggingface.co/spaces/{repo_id}")
