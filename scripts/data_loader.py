# scripts/data_loader.py

import pandas as pd
from pathlib import Path
from scripts.config import METADATA_CSV, IMAGE_DIR

def load_and_clean_metadata():
    """
    Loads the metadata from the CSV file, skips bad lines,
    constructs the full image path, and filters for items
    with existing images.

    Returns:
        pd.DataFrame: A cleaned DataFrame with an 'image_path' column.
    """
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_CSV}. "
            "Please run the dataset download script first."
        )

    try:
        # Load metadata, skipping any broken lines
        meta_df = pd.read_csv(METADATA_CSV, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return pd.DataFrame()

    # Construct the full path for each image
    meta_df['image_path'] = meta_df['id'].astype(str) + ".jpg"
    meta_df['image_path'] = meta_df['image_path'].apply(lambda x: IMAGE_DIR / x)

    # Keep only rows where the image file actually exists
    meta_df = meta_df[meta_df['image_path'].apply(Path.exists)].reset_index(drop=True)

    print(f"Successfully loaded and validated {len(meta_df)} items with images.")
    return meta_df

if __name__ == '__main__':
    # For testing the script directly
    metadata = load_and_clean_metadata()
    print("\nFirst 5 rows of the cleaned metadata:")
    print(metadata.head())