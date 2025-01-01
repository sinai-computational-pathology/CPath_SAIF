import os
import numpy as np
import pandas as pd
import torch
import pdb

def get_datasets(encoder='SP21M', task='classification', class_only=None, data_version='cohort_10_30_2024', organ='skin', shuffle_target=False, random_seed=1234):
    # Load slide data
    tensor_root = '/sc/arion/projects/comppath_SAIF/data/cohort_08_15_2024/'
    master = pd.read_csv(f'/sc/arion/projects/comppath_SAIF/data/{data_version}/slide_master_{organ}.csv')
    
    master['tensor_path'] = [os.path.join(tensor_root, f'{encoder}_features', f'{x.batch:02d}' ,f'{x.slide}.pth') for _, x in master.iterrows()]
    files = master.apply(lambda x: os.path.join('/sc/arion/projects/comppath_500k/datasets/MTL_11_06_23/coordinates/', f'{x.batch:02d}', f'{x.slide}.csv'), axis=1)
    flag = files.apply(os.path.exists)
    master = master[flag]

    target_dict = {
        'White':0,
        'Black':1,
        'Hispanic/Latino':2,
        'Asian':3,
        'Other':4
    }

    master['target'] = master.race_curated.map(target_dict)
    
    # Weights
    weights = 0.2 / master.target.value_counts(normalize=True).values
    weights = torch.FloatTensor(weights)

    # Split into train and val
    df_train = master[master.split=='train'].reset_index(drop=True)
    if shuffle_target:
        df_train['target'] = np.random.RandomState(seed=random_seed).permutation(df_train['target'].values)
    df_val = master[master.split=='val'].reset_index(drop=True)

    if class_only:
        print(f"only use class {class_only} in dataset for training")
        df_train = df_train[df_train['target'] == class_only]

    # Create my loader objects
    if task == "classification":
        dset_train = slide_dataset_classification(df_train)
        dset_val = slide_dataset_classification(df_val)

    elif task == "attention":
        dset_train = slide_dataset_attention(df_train)
        dset_val = slide_dataset_attention(df_val)

    return dset_train, dset_val, weights

# def get_datasets_survival(mccv=0, data='', encoder=''):
#     # Load slide data
#     df = pd.read_csv(os.path.join('/sc/arion/projects/comppath_500k/SSLbenchmarks/data', data, 'slide_data.csv'))
#     df['tensor_path'] = [os.path.join(x.tensor_root, encoder, x.tensor_name) for _, x in df.iterrows()]
#     # Select mccv and clean
#     df = df.rename(columns={'mccv{}'.format(mccv):'mccvsplit'})[['slide','tte', 'event','mccvsplit','tensor_path']]
#     # Split into train and val
#     df_train = df[df.mccvsplit=='train'].reset_index(drop=True).drop(columns=['mccvsplit'])
#     df_val = df[df.mccvsplit=='val'].reset_index(drop=True).drop(columns=['mccvsplit'])
#     # Create my loader objects
#     dset_train = slide_dataset_survival(df_train)
#     dset_val = slide_dataset_survival(df_val)
#     return dset_train, dset_val

class slide_dataset_classification(object):
    '''
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    '''
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        # number of slides
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get the feature matrix for that slide
        h = torch.load(row.tensor_path, weights_only=True)
        # get the target

        return h, row.target
    
class slide_dataset_attention(object):
    '''
    Slide level dataset which returns for each slide the feature matrix (h) and the target
    '''
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        # number of slides
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
        # get the target

        return h

# class slide_dataset_survival(object):
#     '''
#     Slide level dataset which returns for each slide the feature matrix (h) and the target
#     '''
#     def __init__(self, df):
#         self.df = df
    
#     def __len__(self):
#         # number of slides
#         return len(self.df)
    
#     def __getitem__(self, index):
#         row = self.df.iloc[index]
#         # get the feature matrix for that slide
#         h = torch.load(row.tensor_path)
#         # get the target
#         return h, row.tte, row.event
