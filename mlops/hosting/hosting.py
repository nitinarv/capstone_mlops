from huggingface_hub import HfApi
import os

repo_id = f"{os.getenv('YOUR_USERNAME')}/car-engine-predictive-maintenence-model"
repo_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="mlops/deployment",     # the local folder containing your files
    repo_id=repo_id,          # the target repo
    repo_type=repo_type,                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
