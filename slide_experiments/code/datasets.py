import os
import numpy as np
import pandas as pd
import torch

exp_to_data_version = {
    "cohort_08_15_2024": "exp1",
    "cohort_10_30_2024": "exp2",
    "cohort_12_13_2024": "exp3"
}
    
def get_datasets(encoder='SP22M', task='classification', class_only=None, data_version='cohort_10_30_2024', data_sub_version='balanced_cancer', organ='skin', label_type='genetic_ancestry', shuffle_target=None, remove_roi=None, random_seed=1234):
    tensor_root = f'/sc/arion/projects/comppath_SAIF/data/make_features/{encoder}'
    
    # Load master metadata
    master_file = 'master_metadata.csv'
    master_path = f'/sc/arion/projects/comppath_SAIF/data/{organ}/{data_version}/{label_type}/{data_sub_version}/{master_file}'
    master = pd.read_csv(master_path)
    print(f"master:{len(master)}")
    
    # load tile-level roi metadata
    roi_metadata_path = f'/sc/arion/projects/comppath_SAIF/slide_experiments/{organ}/SP22M/exp1/{data_sub_version}/marimo_metadata.csv'
    roi_metadata = pd.read_csv(roi_metadata_path) if os.path.exists(roi_metadata_path) else None
    
    # check if the tensor path exists and if the coordinates exist
    master['slide'] = master.apply(lambda x: f"{x['filename']}_{x['barcode']}", axis=1)
    try:
        master['tensor_path'] = [os.path.join(tensor_root, f'{x.batch:02d}' ,f'{x.slide}.pth') for _, x in master.iterrows()]
        files = master.apply(lambda x: os.path.join('/sc/arion/projects/comppath_500k/datasets/MTL_11_06_23/coordinates/', f'{x.batch:02d}', f'{x.slide}.csv'), axis=1)
        
        tensor_path_exists = master['tensor_path'].apply(os.path.exists)
        coordinates_exist = files.apply(os.path.exists)
        
        dropped_tensor_path = len(master) - tensor_path_exists.sum()
        dropped_coordinates = len(master) - coordinates_exist.sum()
        
        print(f"Rows dropped due to missing tensor_path: {dropped_tensor_path}")
        print(f"Rows dropped due to missing coordinates: {dropped_coordinates}")
        
        # Save rows with missing tensor_path to a CSV
        missing_tensor_path = master[~tensor_path_exists]
        missing_tensor_path.to_csv('missing_tensor_path.csv', index=False)
        print(f"Saved missing tensor_path rows to 'missing_tensor_path.csv'")
        
        flag = tensor_path_exists & coordinates_exist
        master = master[flag]
    except Exception as e:
        print(f"Error occurred while processing tensor paths: {e}")
    print(f"master after checking coordinates:{len(master)}")
    
    # Load slide data
    if label_type == 'genetic_ancestry':
        label_name = 'genetic_determined'
    elif label_type == 'self_reported':
        label_name = 'race_curated'
            
    # Create target mapping dynamically based on unique labels in alphabetical order
    unique_labels = sorted(master[label_name].unique())
    target_dict = {label: idx for idx, label in enumerate(unique_labels)}
    
    master['target'] = master[label_name].map(target_dict)
    
    print(f"master after mapping target:{len(master)}")
    
    # Weights
    weights = 0.2 / master.target.value_counts(normalize=True).values
    weights = torch.FloatTensor(weights)

    # Split into train and val
    df_train = master[master.split=='train'].reset_index(drop=True)
    df_val = master[master.split=='val'].reset_index(drop=True)
    
    print(master.split.value_counts(normalize=True))
    print(f"train:{len(df_train)}, val:{len(df_val)}")
    
    if shuffle_target == 'train':
        df_train['target'] = np.random.RandomState(seed=random_seed).permutation(df_train['target'].values)
    elif shuffle_target == 'val':
        df_val['target'] = np.random.RandomState(seed=random_seed).permutation(df_val['target'].values)

    if class_only:
        print(f"only use class {class_only} in dataset for training")
        df_train = df_train[df_train['target'] == class_only]

    # Create my loader objects
    if task == "classification":
        print(f"Using classification task with remove_roi={remove_roi}")
        dset_train = slide_dataset_classification(df_train, remove_epi=None, roi_metadata=roi_metadata)
        dset_val = slide_dataset_classification(df_val, remove_epi=remove_roi, roi_metadata=roi_metadata)

    elif task == "attention":
        dset_train = slide_dataset_attention(df_train, remove_epi=None, roi_metadata=roi_metadata)
        dset_val = slide_dataset_attention(df_val, remove_epi=remove_roi, roi_metadata=roi_metadata)

    return dset_train, dset_val, weights

class slide_dataset_classification(object):
    '''
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    '''
    def __init__(self, df, remove_epi=None, roi_metadata=None):
        self.df = df
        self.remove_epi = remove_epi
        self.roi_metadata = roi_metadata
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get the feature matrix for that slide
        h = torch.load(row.tensor_path, weights_only=True)
        if self.remove_epi:
            slide_epi_info = self.roi_metadata[self.roi_metadata['slide'] == row.slide].drop_duplicates().reset_index(drop=True)
            mask = slide_epi_info['is_epithelium'].values == 0 if self.remove_epi == "remove" else slide_epi_info['is_epithelium'].values == 1
            if h.shape[0] != slide_epi_info.shape[0]:
                print(f"Shape mismatch for slide {row.slide}: h.shape[0] = {h.shape[0]}, slide_epi_info.shape[0] = {slide_epi_info.shape[0]}")
            assert h.shape[0] == slide_epi_info.shape[0], f"Shape mismatch for slide {row.slide}: h.shape[0] = {h.shape[0]}, slide_epi_info.shape[0] = {slide_epi_info.shape[0]}"
            if mask.any():
                h = h[mask]
        # get the target
        return h, row.target
    
class slide_dataset_attention(object):
    '''
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    '''
    def __init__(self, df, remove_epi=False, roi_metadata=None):
        self.df = df
        self.remove_epi = remove_epi
        self.roi_metadata = roi_metadata
    
    def __len__(self):
        return len(self.df)
    
    def get_tiles(self, index):
        row = self.df.iloc[index]
        tiles = pd.read_csv(os.path.join('/sc/arion/projects/comppath_500k/datasets/MTL_11_06_23/coordinates/', f'{row.batch:02d}', f'{row.slide}.csv'))
        tiles['slide'] = row.slide
        tiles['target'] = row.target
        return tiles
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get the feature matrix for that slide
        h = torch.load(row.tensor_path, weights_only=True)
        if self.remove_epi:
            slide_epi_info = self.roi_metadata[self.roi_metadata['slide'] == row.slide].drop_duplicates().reset_index(drop=True)
            mask = slide_epi_info['is_epithelium'].values == 0 if self.remove_epi == "remove" else slide_epi_info['is_epithelium'].values == 1
            if h.shape[0] != slide_epi_info.shape[0]:
                print(f"Shape mismatch for slide {row.slide}: h.shape[0] = {h.shape[0]}, slide_epi_info.shape[0] = {slide_epi_info.shape[0]}")
            assert h.shape[0] == slide_epi_info.shape[0], f"Shape mismatch for slide {row.slide}: h.shape[0] = {h.shape[0]}, slide_epi_info.shape[0] = {slide_epi_info.shape[0]}"
            if mask.any():
                h = h[mask]
        return h
