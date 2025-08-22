import kagglehub
import shutil
import os

def get_dataset():
    
    cached_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")

    current_dir = os.getcwd()
    dest_path = os.path.join(current_dir, "dataset")

    shutil.copytree(cached_path, dest_path, dirs_exist_ok=True)
    print("Dataset copied to:", dest_path)

if __name__ == "__main__":
    get_dataset()
    print("Dataset is ready for use.")