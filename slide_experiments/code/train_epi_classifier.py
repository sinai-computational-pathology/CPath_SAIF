import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
import pickle  
import glob
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

def load_ground_truth(ground_truth_path):
    """Load ground truth data from CSV files."""
    all_files = glob.glob(os.path.join(ground_truth_path, "*.csv"))
    df_list = [pd.read_csv(file) for file in all_files]
    ground_truth_df = pd.concat(df_list, ignore_index=True)
    
    # Check and fill NA values in the 'annotation' column
    ground_truth_df['annotation'] = ground_truth_df['annotation'].fillna("N/A")
    
    # Print value counts of the 'annotation' column
    print(ground_truth_df['annotation'].value_counts())
    
    return ground_truth_df

def apply_transform(meta_df, pca_model_path, umap_model_path, output_path, batch_index, batch_size=50):
    """Apply pretrained PCA model to reduce dimensions to 2 features and save to file."""

    output_features_df_path = Path(output_path) / f"batch_{batch_index}.csv"
    if output_features_df_path.exists():
        print(f"Batch {batch_index} already processed. Skipping.")
        return
    
    pca_model = joblib.load(pca_model_path)
    # Print cumulative variance explained by the first 2 principal components
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)[:2]
    print(f"Cumulative variance explained by the first 2 principal components: {cumulative_variance[-1]:.2f}")
    
    umap_model = joblib.load(umap_model_path)
    
    slides = meta_df['slide'].unique()
    start_idx = (batch_index - 1) * batch_size
    end_idx = start_idx + batch_size
    batch_slides = slides[start_idx:end_idx]
    
    umap_features_list = []

    for slide in tqdm(batch_slides, desc="Loading and transforming embeddings"):
        slide_df = meta_df[meta_df['slide'] == slide].drop_duplicates(subset=['slide', 'x', 'y'])
        tensor_path = slide_df['tensor_path'].iloc[0]
        embedding = torch.load(tensor_path, weights_only=True)
        pca_features = pca_model.transform(embedding)
        umap_features = umap_model.transform(pca_features)
        
        slide_df[['UMAP_1', 'UMAP_2']] = umap_features[:, :2]
        umap_features_list.append(slide_df[['slide', 'x', 'y', 'UMAP_1', 'UMAP_2']])
        del pca_features  # Remove temporary pca_features to save memory

    # Concatenate all PCA features and save to a .npy file
    umap_features_df = pd.concat(umap_features_list, ignore_index=True)
    umap_features_df.to_csv(output_features_df_path, index=False)

def predict_is_epithelium(meta_df, trained_model, filter_threshold=None):
    """Predict 'is_epithelium' for all slides in meta_df."""
    if trained_model is not None:
        X_all = meta_df[['UMAP_1','UMAP_2']].values  # Use only the top 2 UMAP components
        meta_df['is_epithelium'] = trained_model.predict(X_all)
    else:
        raise ValueError("Need to provide a trained model for inference")

    if filter_threshold is not None:
        epithelium_counts = meta_df.groupby('slide')['is_epithelium'].sum()
        valid_slides = epithelium_counts[epithelium_counts >= filter_threshold].index
        removed_slides = set(meta_df['slide'].unique()) - set(valid_slides)
        meta_df = meta_df[meta_df['slide'].isin(valid_slides)]
        print(f"Number of slides removed due to not satisfying the threshold: {len(removed_slides)}")
    else:
        print("No threshold set")

    return meta_df.reset_index(drop=True)

def classify_epithelium(df, method="logistic", test_size=0.4, random_state=42):
    if method == "logistic":
        model = LogisticRegression(random_state=random_state)
    elif method == "random_forest":
        model = RandomForestClassifier(random_state=random_state)
    else:
        raise ValueError("Invalid method. Choose 'logistic' or 'random_forest'.")

    # Filter labeled patches
    labeled_patches = df[df['annotation'].notna()]
    positive_samples = labeled_patches[labeled_patches['annotation'].str.lower() == "epithelium"]

    # Select negative samples (unlabeled patches)
    negative_samples = df[df['annotation']=='non-epithelium']
    if len(negative_samples) < len(positive_samples):
        print("Not enough negative samples, using all available negative samples.")
    else:
        negative_samples = negative_samples.sample(n=len(positive_samples), random_state=random_state)

    # Combine positive and negative samples
    training_data = pd.concat([positive_samples, negative_samples])
    X = training_data[['UMAP_D1','UMAP_D2']].values  # Use only the top 2 UMAP
    y = (training_data['annotation'].str.lower() == "epithelium").fillna(False).astype(int)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Epithelium", "Epithelium"]))

    # # Classify all patches using the model
    # X_all = meta_df[['UMAP_D1','UMAP_D2']].values  # Use only the top 2 UMAP
    # meta_df['is_epithelium'] = model.predict(X_all)

    # Save the trained model
    today_date = datetime.now().strftime("%Y-%m-%d")
    model_filename = f"/sc/arion/projects/comppath_SAIF/slide_experiments/skin/SP22M/exp3/model_{method}_{today_date}.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

    return model

def prepare_metadata(master_df, subset_split='all'):
    """
    Prepares metadata for slides, coordinates, and embeddings.

    Args:
        add_slide_metadata (list): List of DataFrames for additional slide-level metadata.
        add_tile_metadata (list): List of DataFrames for additional tile-level metadata.

    Returns:
        pd.DataFrame: Final metadata with tile-level information, embeddings paths, and additional metadata.
    """
    # Step 1: Load and filter master_df (slide-level information)
    print("Loading master file...")

    if subset_split != "all":
        print(f"Filtering master file for subset split: {subset_split}")
        master_df = master_df[master_df.split == subset_split].reset_index(drop=True)
    else:
        print("No subset split specified, using all slides.")

    print("Checking slide existence...")
    flag = master_df['dataark_path'].apply(os.path.exists)
    master_df = master_df[flag].reset_index(drop=True)

    # Step 2: Load and consolidate coordination (tile-level information)
    print("Loading coordinates...")
    tile_metadata = []
    valid_slides = set()
    for _, row in master_df.iterrows():
        coord_path = os.path.join(args.coordinate_path, f"{row.batch:02d}", f"{row.slide}.csv")
        if os.path.exists(coord_path):
            coords = pd.read_csv(coord_path)
            coords['slide'] = row.slide 
            tile_metadata.append(coords)
            valid_slides.add(row.slide)
        else:
            print(f"Coordinate file missing for slide: {row.slide}")

    if not tile_metadata:
        raise ValueError("No valid coordinate files found.")
    coordination = pd.concat(tile_metadata, ignore_index=True)

    # Filter master_df to only include slides with valid coordinate files
    master_df = master_df[master_df['slide'].isin(valid_slides)].reset_index(drop=True)
    print(f"Number of slides with valid coordinate files: {len(master_df)}")

    # Step 3: Repeat slide-level metadata for each tile and merge
    print("Merging slide-level information with tile-level metadata...")
    meta_df = coordination.merge(master_df, on="slide", how="left")

    # Step 4: Add embedding paths
    print("Adding embedding paths...")
    meta_df['tensor_path'] = [
        os.path.join(args.embedding_path, f"{row.batch:02d}", f"{row.slide}.pth")
        for _, row in meta_df.iterrows()
    ]

    # Ensure only rows with valid embedding paths are retained
    meta_df = meta_df[meta_df['tensor_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"Number of tiles with valid embeddings: {len(meta_df)}")
    
    return meta_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform New Data Using Pretrained Models")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for results")
    parser.add_argument("--batch_index", type=int, required=True, help="Batch index for processing")
    parser.add_argument("--coordinate_path", type=str, required=True, help="Path to the coordinate files")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to the embedding tensorpath")
    parser.add_argument("--pca_model_path", type=str, required=True, help="Path to the pretrained PCA model")
    parser.add_argument("--umap_model_path", type=str, required=True, help="Path to the pretrained UMAP model")
    parser.add_argument("--organ", type=str, required=True, help="Organ type")
    parser.add_argument("--encoder", type=str, required=True, help="Encoder type")
    parser.add_argument("--exp_version", type=str, required=True, help="Experiment version")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to the ground truth CSV files")
    
    args = parser.parse_args()

    # Identify validation slides from the three CSV files and process experiments
    exp_to_data_version = {
        "exp1": "cohort_08_15_2024",
        "exp2": "cohort_10_30_2024",
        "exp3": "cohort_12_13_2024"
    }
    
    organ = args.organ
    encoder = args.encoder
    exp_version = args.exp_version
    data_version = exp_to_data_version[exp_version]
    
    output_path = Path(args.output_path) 
    output_path.mkdir(parents=True, exist_ok=True)
    output_meta_path = os.path.join(output_path, f"meta_df_all.csv")
    
    # Load master metadata
    master_metadata_path = f'/sc/arion/projects/comppath_SAIF/data/skin/{data_version}/master_metadata.csv'
    master_metadata_df = pd.read_csv(master_metadata_path)
    
    if not os.path.exists(output_meta_path):
        meta_df = prepare_metadata(master_metadata_df)
        meta_df.to_csv(output_meta_path, index=False)
        print(f"Saving generated meta_df to {output_meta_path}...")
    else:
        meta_df = pd.read_csv(output_meta_path)
        print(f"Loading existing meta_df from {output_meta_path}...")
    
    # Calculate the number of expected batch CSV files
    num_batches = len(meta_df['slide'].unique()) // 50 + 1
    print(f"Number of expected batch CSV files: {num_batches}")

    # Check if all batch CSV files exist
    batch_files = [output_path / f"batch_{i+1}.csv" for i in range(num_batches)]
    
    apply_transform(meta_df, args.pca_model_path, args.umap_model_path, output_path, args.batch_index)
    
    if all(batch_file.exists() for batch_file in batch_files):
        print("All batch CSV files already exist. Proceeding to classify epithelium.")
        
        # Load and merge all batch CSV files
        batch_dfs = [pd.read_csv(batch_file) for batch_file in batch_files]
        merged_meta_df = pd.concat(batch_dfs, ignore_index=True)
        
        # Load ground truth data
        ground_truth_df = load_ground_truth(args.ground_truth_path)
        
        # Classify epithelium using logistic regression
        trained_model = classify_epithelium(ground_truth_df, method='random_forest')

        # Predict is_epithelium for slides
        meta_df = predict_is_epithelium(merged_meta_df, trained_model)

        # Save the updated meta_df with selected columns
        selected_columns = ['slide', 'x', 'y', 'is_epithelium']
        meta_df[selected_columns].to_csv(Path(output_path) / "for_remove_epi_exp3.csv", index=False)
        print(f"Updated meta_df saved to {Path(output_path) / 'for_remove_epi_exp3.csv'}")

