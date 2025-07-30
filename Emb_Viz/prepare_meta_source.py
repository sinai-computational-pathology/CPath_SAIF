import os
import sys
import pandas as pd
import numpy as np
import joblib
import umap
from sklearn.decomposition import IncrementalPCA
from pathlib import Path
from tqdm import tqdm
import argparse
import torch

class DimensionalityReducer:
    def __init__(self, meta_df, mode='PCA+UMAP', n_pca_components=50, n_umap_components=2):
        """
        Initialize the DimensionalityReducer.

        Args:
            meta_df (pd.DataFrame): Metadata containing slide-level information.
            output_path (str): Path to save results.
            mode (str): Processing mode ('PCA', 'UMAP', 'PCA+UMAP').
            batch_size (int): Number of tiles to process in a single batch for transformation.
            tile_limit (int): Maximum number of tiles to load for each PCA partial fit step.
            n_pca_components (int): Number of PCA components.
            n_umap_components (int): Number of UMAP components.
        """
        self.meta_df = meta_df
        self.output_path = Path(args.output_path)
        self.mode = mode
        self.batch_size = args.batch_size
        self.tile_limit = args.tile_limit
        self.n_pca_components = n_pca_components
        self.n_umap_components = n_umap_components

    def load_batch_embeddings(self, tensor_paths, tile_limit):
        """
        Load embeddings in batches.

        Args:
            tensor_paths (list): List of paths to tensor files.
            tile_limit (int): Maximum number of tiles to load before yielding a batch.

        Yields:
            np.array: Batch of embeddings.
        """
        accumulated_embeddings = []
        accumulated_tiles = 0
        
        for tensor_path in tqdm(tensor_paths, desc="Loading embeddings in batches"):
            embedding = torch.load(tensor_path, weights_only=True)
            accumulated_embeddings.append(embedding)
            accumulated_tiles += embedding.shape[0]

            if accumulated_tiles >= tile_limit:
                yield np.vstack(accumulated_embeddings)
                accumulated_embeddings = []
                accumulated_tiles = 0

        if accumulated_embeddings:
            yield np.vstack(accumulated_embeddings)

    def fit_pca(self, tensor_paths):
        """
        Fit PCA incrementally on embeddings.

        Args:
            tensor_paths (list): List of paths to embedding files.

        Returns:
            IncrementalPCA: Fitted PCA model.
        """
        print("Fitting PCA incrementally...")
        pca = IncrementalPCA(n_components=self.n_pca_components)

        for batch in self.load_batch_embeddings(tensor_paths, self.tile_limit):
            pca.partial_fit(batch)

        # Save PCA model
        pca_model_path = self.output_path / f"pca{self.n_pca_components}_model.joblib"
        joblib.dump(pca, pca_model_path)
        print(f"PCA model saved to {pca_model_path}")
        return pca

    def transform_pca(self, tensor_paths, pca):
        """
        Transform embeddings using a fitted PCA model.

        Args:
            tensor_paths (list): List of paths to embedding files.
            pca (IncrementalPCA): Fitted PCA model.

        Returns:
            np.array: PCA-transformed features.
        """
        print("Transforming embeddings using PCA...")
        pca_features = []

        for batch in self.load_batch_embeddings(tensor_paths, self.tile_limit):
            pca_features.append(pca.transform(batch))

        pca_features = np.vstack(pca_features)

        # Save PCA features
        pca_features_path = self.output_path / "pca_features.npy"
        np.save(pca_features_path, pca_features)
        print(f"PCA features saved to {pca_features_path}")

        return pca_features

    def fit_umap(self, pca_features):
        """
        Fit UMAP on PCA-transformed features.

        Args:
            pca_features (np.array): PCA-transformed features.

        Returns:
            UMAP: Fitted UMAP model.
        """
        umap_model_path = self.output_path / "umap_model.joblib"
        if umap_model_path.exists():
            print(f"Loading existing UMAP model from {umap_model_path}...")
            reducer = joblib.load(umap_model_path)
        else:
            print("Fitting UMAP...")
            if pca_features.shape[0] > 20000000:
                print(f"Subsampling PCA features from {pca_features.shape[0]} to 20000000...")
                indices = np.random.choice(pca_features.shape[0], 20000000, replace=False)
                pca_features = pca_features[indices]
            reducer = umap.UMAP(n_components=self.n_umap_components, low_memory=True)
            reducer.fit(pca_features)
            joblib.dump(reducer, umap_model_path)
            print(f"UMAP model saved to {umap_model_path}")
        return reducer

    def transform_umap_in_batches(self, pca_features, reducer):
        """
        Transform PCA-transformed features using a fitted UMAP model in batches.

        Args:
            pca_features (np.array): PCA-transformed features.
            reducer (UMAP): Fitted UMAP model.

        Returns:
            np.array: UMAP-transformed features.
        """
        print("Transforming PCA features using UMAP in batches...")
        umap_features = []

        for i in tqdm(range(0, len(pca_features), self.batch_size), desc="Transforming UMAP", file=sys.stdout):
            batch = pca_features[i:i + self.batch_size]
            umap_features.append(reducer.transform(batch))
            # Release memory
            del batch
            torch.cuda.empty_cache()

        # Concatenate all UMAP features
        umap_features = np.vstack(umap_features)

        # Save final UMAP features
        umap_features_path = self.output_path / "umap_features.npy"
        np.save(umap_features_path, umap_features)
        print(f"UMAP features saved to {umap_features_path}")

        return umap_features


    def forward(self, tensor_paths, val_tensor_paths=None):
        """
        Execute the dimensionality reduction pipeline.

        Args:
            tensor_paths (list): List of paths to tensor files.
            val_tensor_paths (list): List of paths to tensor files for validation set (optional).

        Returns:
            None
        """
        pca_features_path = self.output_path / "pca_features.npy"
        umap_features_path = self.output_path / "umap_features.npy"

        # Check if PCA features already exist
        if pca_features_path.exists():
            print(f"Loading existing PCA features from {pca_features_path}...")
            pca_features = np.load(pca_features_path)
            # Load PCA model
            pca_model_path = self.output_path / f"pca{self.n_pca_components}_model.joblib"
            if pca_model_path.exists():
                print(f"Loading existing PCA model from {pca_model_path}...")
                pca_model = joblib.load(pca_model_path)
            else:
                raise ValueError("PCA model not found. Please ensure the PCA model exists.")
        else:
            # Check if PCA model already exists
            pca_model_path = self.output_path / f"pca{self.n_pca_components}_model.joblib"
            if pca_model_path.exists():
                print(f"Loading existing PCA model from {pca_model_path}...")
                pca_model = joblib.load(pca_model_path)
            else:
                # PCA Step
                if 'PCA' in self.mode:
                    pca_model = self.fit_pca(tensor_paths)
                else:
                    raise ValueError("PCA is required before UMAP. Please include 'PCA' in mode.")
            pca_features = self.transform_pca(tensor_paths, pca_model)

        # Check if UMAP features already exist
        if 'UMAP' in self.mode:
            if umap_features_path.exists():
                print(f"Loading existing UMAP features from {umap_features_path}...")
                umap_features = np.load(umap_features_path)
            else:
                # Check if UMAP model already exists
                umap_model_path = self.output_path / "umap_model.joblib"
                if umap_model_path.exists():
                    print(f"Loading existing UMAP model from {umap_model_path}...")
                    umap_model = joblib.load(umap_model_path)
                else:
                    # UMAP Step
                    umap_model = self.fit_umap(pca_features)
                if val_tensor_paths:
                    val_pca_features = self.transform_pca(val_tensor_paths, pca_model)
                    umap_features = self.transform_umap_in_batches(val_pca_features, umap_model)
                else:
                    umap_features = self.transform_umap_in_batches(pca_features, umap_model)
        else:
            umap_features = None

        # Save final data
        self.save_final_data(self.meta_df, pca_features, umap_features)

    def save_final_data(self, meta_df, pca_features, umap_features):
        """
        Combine metadata with PCA and UMAP features and save the final data.

        Args:
            meta_df (pd.DataFrame): Metadata DataFrame.
            pca_features (np.array): PCA-transformed features.
            umap_features (np.array): UMAP-transformed features.

        Returns:
            None
        """
        print("Saving final transformed metadata...")
        meta_df = meta_df.copy()
        # meta_df["PCA50"] = list(pca_features)
        meta_df["PC1"] = pca_features[:, 0]
        meta_df["PC2"] = pca_features[:, 1]
        if umap_features is not None:
            meta_df["UMAP_D1"] = umap_features[:, 0]
            meta_df["UMAP_D2"] = umap_features[:, 1]

        # Reorder columns and save
        columns_to_keep = ['slide', 'slidepath', 'race_curated', 'x', 'y', 'PC1', 'PC2'] # need to modify for different scenarios
        if umap_features is not None:
            columns_to_keep += ['UMAP_D1', 'UMAP_D2']
        meta_df = meta_df[columns_to_keep + ['tensor_path']]

        final_data_path = self.output_path / "marimo_metadata.csv"
        meta_df.to_csv(final_data_path, index=False)
        print(f"Final transformed metadata saved to {final_data_path}")

def prepare_metadata(add_patient_metadata=[], add_case_metadata=[], add_slide_metadata=[], add_tile_metadata=[]):
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
    master_df = pd.read_csv(args.master_file)
    if args.subset_split != "all":
        print(f"Filtering master file for subset split: {args.subset_split}")
        master_df = master_df[master_df.split == args.subset_split].reset_index(drop=True)
    else:
        print("No subset split specified, using all slides.")

    print("Checking slide existence...")
    flag = master_df['slide_path'].apply(os.path.exists)
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

    # Step 5: Merge and rename additional patient-level metadata
    if add_patient_metadata:
        print("Merging additional patient-level metadata (MRN)...")
        for df in add_patient_metadata:
            if 'MRN' not in df.columns:
                raise ValueError("Patient-level metadata must contain an 'MRN' column.")
            
            # Rename columns to include "patient-" prefix
            df_renamed = df.rename(columns={col: f"patient-{col}" for col in df.columns if col != 'MRN'})
            meta_df = meta_df.merge(df_renamed, on="MRN", how="left")

    # Step 6: Merge and rename additional case-level metadata
    if add_case_metadata:
        print("Merging additional case-level metadata (accession_no)...")
        for df in add_case_metadata:
            if 'accession_no' not in df.columns:
                raise ValueError("Case-level metadata must contain an 'accession_no' column.")
            
            # Rename columns to include "case-" prefix
            df_renamed = df.rename(columns={col: f"case-{col}" for col in df.columns if col != 'accession_no'})
            meta_df = meta_df.merge(df_renamed, on="accession_no", how="left")

    # Step 7: Merge and rename additional slide-level metadata
    if add_slide_metadata:
        print("Merging additional slide-level metadata (slide)...")
        for df in add_slide_metadata:
            if 'slide' not in df.columns:
                raise ValueError("Slide-level metadata must contain a 'slide' column.")
            
            # Rename columns to include "slide-" prefix
            df_renamed = df.rename(columns={col: f"slide-{col}" for col in df.columns if col != 'slide'})
            meta_df = meta_df.merge(df_renamed, on="slide", how="left")

    # Step 8: Merge additional tile-level metadata
    if add_tile_metadata:
        print("Merging additional tile-level metadata...")
        for df in add_tile_metadata:
            if not set(['slide', 'coordinate_x', 'coordinate_y']).issubset(df.columns):
                raise ValueError("Tile-level metadata must contain 'slide', 'coordinate_x', and 'coordinate_y' columns.")
            meta_df = meta_df.merge(df, on=["slide", "coordinate_x", "coordinate_y"], how="left")

    print(f"Final meta_df have shape of {meta_df.shape} with columns of {meta_df.columns}")
    print(meta_df.head())

    return meta_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General Metadata Preparation and Dimension Reduction Pipeline")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for results")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to the metadata file (CSV)")
    parser.add_argument("--coordinate_path", type=str, required=True, help="Path to coordinate files")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to embedding files")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--tile_limit", type=int, default=400000, help="Tile limit for PCA partial fitting")
    parser.add_argument("--meta_df_name", type=str, default="marimo_metadata", help="Name of output metadata CSV file")
    parser.add_argument("--mode", type=str, default="PCA+UMAP", help="Dimensionality reduction mode (PCA, UMAP, PCA+UMAP)")
    global args
    args = parser.parse_args()

    # Load metadata
    print("Loading metadata file...")
    master_df = pd.read_csv(args.metadata_file)

    # Check slide existence
    print("Checking slide existence...")
    flag = master_df['slide_path'].apply(os.path.exists)
    master_df = master_df[flag].reset_index(drop=True)

    # Load and consolidate coordinates
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

    # Merge slide-level information with tile-level metadata
    print("Merging slide-level information with tile-level metadata...")
    meta_df = coordination.merge(master_df, on="slide", how="left")

    # Add embedding paths
    print("Adding embedding paths...")
    meta_df['tensor_path'] = [
        os.path.join(args.embedding_path, f"{row.slide}.pth")
        for _, row in meta_df.iterrows()
    ]
    meta_df = meta_df[meta_df['tensor_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"Number of tiles with valid embeddings: {len(meta_df)}")

    # Save generated metadata
    output_meta_path = os.path.join(args.output_path, f"{args.meta_df_name}.csv")
    meta_df.to_csv(output_meta_path, index=False)
    print(f"Saving generated meta_df to {output_meta_path}...")

    # Run dimensionality reduction
    reducer = DimensionalityReducer(meta_df, mode=args.mode)
    tensor_paths = meta_df["tensor_path"].drop_duplicates().tolist()
    reducer.forward(tensor_paths)