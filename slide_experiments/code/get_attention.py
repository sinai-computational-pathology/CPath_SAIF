import datasets
import modules
import os
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import pdb
from sklearn.metrics import roc_auc_score
import pickle

parser = argparse.ArgumentParser()

#I/O PARAMS
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--checkpoint_path', default='', type=str, help='path to checkpoint')
parser.add_argument('--encoder', type=str, default='', choices=[
    'SP21M',
    'SP85M',
    'uni',
    'virchow',
    'virchow2',
    'gigapath',
], help='which encoder to use')
parser.add_argument('--aggregator', default='GMA', type=str, help='aggregator to use')
#OPTIMIZATION PARAMS
parser.add_argument('--data_version', required=True, type=str, help="The dataset version to use")
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')


def main():
    
    # Get user input
    global args
    args = parser.parse_args()

    # Set datasets
    _, val_dset, _ = datasets.get_datasets(encoder=args.encoder, task='attention', data_version=args.data_version)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=args.workers)
    
    # Dim of features
    if args.encoder == 'ctranspath':
        args.ndim = 768
    elif args.encoder == 'phikon':
        args.ndim = 768
    elif args.encoder == 'uni':
        args.ndim = 1024
    elif args.encoder == 'virchow':
        args.ndim = 2560
    elif args.encoder == 'virchow2':
        args.ndim = 2560
    elif args.encoder == 'gigapath':
        args.ndim = 1536
    elif args.encoder == 'h-optimus-0':
        args.ndim = 1536
    elif args.encoder == 'SP21M':
        args.ndim = 384
    elif args.encoder == 'SP85M':
        args.ndim = 768
    
    # Get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.aggregator == 'GMA':
        model = modules.GMA(dropout=True, n_classes=5, n_tasks=1, ndim=args.ndim)
    elif args.aggregator == 'GMA_multiple':
        model = modules.GMA_multiple(dropout=True, n_classes=5, n_tasks=5, ndim=args.ndim)
    else:
        raise ("aggregator is not available")
    model.to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    attention_scores_dict, slide_output = test(val_loader, model, device)

    with open(os.path.join(args.output,'attention_scores_dict.pkl'), 'wb') as f:
        pickle.dump(attention_scores_dict, f)

    slide_output.to_csv(os.path.join(args.output, "output_probs.csv"), index=False)
    print(f'All batches processed and saved to CSV at {args.output}.')

def test(loader, model, device):
    # Set model in evaluation mode
    model.eval()

    attention_scores_dict = {}  # To store attention scores per slide with slide_id as key
    all_probs = []              # To store probabilities for each slide
    all_slide_info = []         # To store slide-level info (e.g., labels)

    with torch.no_grad():
        for i, input in enumerate(loader):
            input = input.squeeze(0).to(device)  # Shape: [num_tiles, feature_dim]
            target = loader.dataset.df.iloc[i]['target']
            slide = loader.dataset.df.iloc[i]['slide']

            # Get attention scores
            attention_scores = model(input, attention_only=True)  # Shape: [1, num_tiles] or [5, num_tiles]

            # Get class probabilities
            logits = model(input, attention_only=False)
            logits = logits.cpu().squeeze(0)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

            # Save attention scores in a dictionary with slide_id as the key
            attention_scores_dict[slide] = attention_scores.T  # Shape: [num_tiles, 5]

            all_probs.append(probs)  # Slide-level probabilities
            all_slide_info.append({'slide': slide, 'target': target})

            print(f'Processed slide {i+1}/{len(loader)}')

    # Create DataFrame for slide-level probabilities
    slide_probs_df = pd.DataFrame(all_probs, columns=['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other'])
    slide_info_df = pd.DataFrame(all_slide_info)
    slide_output = pd.concat([slide_info_df, slide_probs_df], axis=1)

    return attention_scores_dict, slide_output

if __name__ == '__main__':
    main()
