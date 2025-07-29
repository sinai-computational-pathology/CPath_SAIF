import os
import numpy as np
import pandas as pd
import argparse
import openslide
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image
from SlideTileExtractor import extract_tissue
import encoders

class slide_dataset(data.Dataset):
    def __init__(self, slide, df, trans, tilesize):
        self.slide = slide
        self.df = df
        self.tilesize = tilesize
        self.trans = trans
    def __getitem__(self, index):
        row = self.df.iloc[index]
        size = int(np.round(self.tilesize * row.mult))
        img = self.slide.read_region((int(row.x), int(row.y)), int(row.level), (size, size)).convert('RGB')
        if row.mult != 1:
            img = img.resize((self.tilesize, self.tilesize), Image.LANCZOS)
        img = self.trans(img)
        return img
    def __len__(self):
        return len(self.df)

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='uni', help='encoder to use')
parser.add_argument('--tilesize', type=int, default=224, help='tile size')
parser.add_argument('--bsize', type=int, default=512, help='batchs size')
parser.add_argument('--workers', type=int, default=10, help='workers')
parser.add_argument('--meta_data_csv', type=str, required=True, default=None, help='meta data csv file for generating csv')

def main():
    global args
    args = parser.parse_args()
    
    # Set up encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, ndim = encoders.get_encoder(args.encoder)
    model.to(device)
    
    # Load metadata
    df = pd.read_csv(args.meta_data_csv)

    output_directory = f'{args.encoder}/features'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    with torch.no_grad():
        for i, row in df.iterrows():
            print(f'[{i+1}]/[{len(df)}]', end='\r')

            # Always generate coordinates
            slide = openslide.OpenSlide(row.slide_path)
            grid = extract_tissue.make_sample_grid(slide, patch_size=args.tilesize, mpp=getattr(args, 'mpp', None), overlap=1, mult=4)
            grid = pd.DataFrame(np.array(grid), columns=['x', 'y']).astype(int)
            base_mpp = None
            level, mult = extract_tissue.find_level(slide, 0.5, patchsize=args.tilesize, base_mpp=base_mpp)
            grid['level'] = level
            grid['mult'] = mult

            # Save grid to CSV in output_directory/coordinate
            coord_dir = os.path.join(output_directory, 'coordinates')
            os.makedirs(coord_dir, exist_ok=True)
            grid.to_csv(os.path.join(coord_dir, f'{row.slide}.csv'), index=False)

            # Output name
            tensor_name = f'{row.slide}.pth'
            if not os.path.exists(os.path.join(output_directory, tensor_name)):
                print(f"file {tensor_name} not exists, generating")
                # Set up dataset and loader
                dset = slide_dataset(slide, grid, transform, args.tilesize)
                loader = torch.utils.data.DataLoader(dset, batch_size=args.bsize, shuffle=False, num_workers=args.workers)
                # Save tensor
                tensor = torch.zeros(len(grid), ndim).float()
                for j, img in enumerate(loader):
                    out = model(img.cuda())
                    tensor[j*args.bsize:j*args.bsize+img.size(0),:] = out.detach().clone()
                torch.save(tensor, os.path.join(output_directory, tensor_name))
    
    print('')
    
if __name__ == '__main__':
    main()
