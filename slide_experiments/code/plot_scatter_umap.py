import os
import numpy as np
import pandas as pd
import argparse
import openslide
from PIL import Image
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='path to data file')
parser.add_argument('--sample', type=int, default=0, help='Sample dataset if too big. Default: 0')
parser.add_argument('--ncol', type=int, default=20, help='Number of images per col. Default: 20')
parser.add_argument('--nrow', type=int, default=20, help='Number of images per row. Default: 20')
parser.add_argument('--output', type=str, default='', help='Path to output file')
parser.add_argument('--alpha', action='store_true', help='to add alpha transparency to encode density')
parser.add_argument('--proj1', type=str, default='C1', help='column of projetion coordinate 1')
parser.add_argument('--proj2', type=str, default='C2', help='column of projetion coordinate 2')
parser.add_argument('--barcode', type=str, default='barcode', help='column of barcode')
parser.add_argument('--x', type=str, default='x', help='column of x')
parser.add_argument('--y', type=str, default='y', help='column of y')
parser.add_argument('--ncell', type=int, default=16, help='number of cells per block')
parser.add_argument('--mincells', type=int, default=100, help='minimum number of cells per block to consider block valid. Default: 100')
parser.add_argument('--cellsize', type=int, default=32, help='cell size')
parser.add_argument('--mpp', type=float, default=0.25, help='MPP')
parser.add_argument('--dpi', type=int, default=100, help='cell size')
parser.add_argument('--hue', type=str, default='', help='hue variable. Default: None')
parser.add_argument('--text', type=str, default='', help='text variable. Default: None')
parser.add_argument('--extent', type=float, nargs='*', default=[], help='if given, the subregion of projection space that will be plotted. Default: None')
parser.add_argument('--showcoords', type=int, default=0, choices=[0, 1], help='to show coords in projection space of each block')

def get_grid(extent, nrow, ncol):
    '''
    extent: xmin, xmax, ymin, ymax
    '''
    e = 1e-5
    xo, xf, yo, yf = extent
    xo, yo = xo-e, yo-e
    xf, yf = xf+e, yf+e
    grid_x = np.linspace(xo, xf, ncol+1)
    grid_y = np.linspace(yo, yf, nrow+1)
    start_x, start_y = np.meshgrid(grid_x[0:-1], grid_y[0:-1])
    end_x, end_y = np.meshgrid(grid_x[1:], grid_y[1:])
    grid = list(zip(start_x.flatten(), end_x.flatten(), start_y.flatten(), end_y.flatten()))
    grid = pd.DataFrame(grid, columns=['xo','xf','yo','yf'])
    return grid

def assign_to_grid(sx, sy, txo, txf, tyo, tyf):
    '''
    sx, sy: 1-d array source projection
    txo, txf, tyo, tyf: 1-d array target x/y min/max limits
    '''
    xgt = sx.reshape(-1,1) >= txo.reshape(1,-1)
    xlt = sx.reshape(-1,1) < txf.reshape(1,-1)
    ygt = sy.reshape(-1,1) >= tyo.reshape(1,-1)
    ylt = sy.reshape(-1,1) < tyf.reshape(1,-1)
    assig = xgt * xlt * ygt * ylt
    counts = assig.sum(0)
    dens = assig.sum(0) / len(sx)
    dens = dens / dens.max()
    dens = pd.DataFrame({'block':np.arange(len(txo)), 'counts': counts, 'density':dens})
    assig = assig.nonzero()[1]
    assig = pd.DataFrame({'block':assig})
    return assig, dens

def get_slide_path(barcode):
    root = '/sc/arion/projects/pathologywsi/tiff_archive/'
    master = pd.read_csv('/sc/arion/projects/comppath_liver/cellMIL/data/prostate_training_master_dataark.csv')
    #print(master.loc[master.barcode==barcode])
    row = master.loc[master.barcode==barcode].iloc[0]
    path = row.slide_path
    return path

def get_cells(df, mpp, cellsize):
    if mpp == 0.5:
        level = 1
    elif mpp == 0.25:
        level = 0
    imgs = []
    for i, row in df.iterrows():
        print(f'{i+1}/{len(df)}', end='\r')
        slidepath = get_slide_path(row.barcode)
        slide = openslide.OpenSlide(slidepath)
        try:
            imgs.append(slide.read_region((int(row.x), int(row.y)), level, (cellsize, cellsize)).convert('RGB'))
        except:
            print(row)
    return imgs

def list_to_mosaic(imgs):
    w = imgs[0].size[0]
    s = int(np.sqrt(len(imgs)))
    S = w * s
    dst = Image.new('RGB', (S, S))
    for i in range(len(imgs)):
        x = i//s * w
        y = i%s * w
        dst.paste(imgs[i], (x, y))
    return dst

def main():
    global args
    args = parser.parse_args()
    
    # Round ncells to closest square
    args.ncell = int(np.round(np.sqrt(args.ncell))**2)
    
    # Read data
    df = pd.read_csv(args.data)
    master = pd.read_csv('/sc/arion/projects/comppath_liver/cellMIL/data/prostate_training_master_dataark.csv')
    df = df[df.barcode.isin(master.barcode)].reset_index(drop=True)
    
    # Calculate extent
    if not args.extent:
        xo, xf = df[args.proj1].min(), df[args.proj1].max()
        yo, yf = df[args.proj2].min(), df[args.proj2].max()
        extent = (xo, xf, yo, yf)
    else:
        extent = (args.extent[0], args.extent[1], args.extent[2], args.extent[3])
        df = df[
            (df[args.proj1]>=args.extent[0]) &
            (df[args.proj1]<=args.extent[1]) &
            (df[args.proj2]>=args.extent[2]) &
            (df[args.proj2]<=args.extent[3])
        ].reset_index(drop=True)
    
    # Definition of grid
    grid = get_grid(extent, args.nrow, args.ncol)
    # Assignment to grid blocks
    assig, dens = assign_to_grid(df[args.proj1].values, df[args.proj2].values, grid.xo.values, grid.xf.values, grid.yo.values, grid.yf.values)
    df = pd.concat([df, assig], axis=1)
    grid = pd.concat([grid, dens], axis=1)
    
    # Plotting
    figw = args.ncol * args.cellsize * int(np.sqrt(args.ncell)) / args.dpi
    figh = args.nrow * args.cellsize * int(np.sqrt(args.ncell)) / args.dpi
    plt.figure(figsize=(figw, figh))
    gs = gridspec.GridSpec(
        nrows=args.nrow, ncols=args.ncol, left=0.1, bottom=0.1, right=0.9, top=0.9,
        wspace=0., hspace=0., width_ratios=[1]*args.ncol, height_ratios=[1]*args.nrow
    )
    for i, row in grid.iterrows():
        if row['counts'] > args.mincells:
            print(f'Block [{i+1}]/[{len(grid)}]')
            # Subset of points
            tmp = df[df.block==row.block]
            tmp = tmp.sample(n=args.ncell).reset_index(drop=True)
            # Get list of images
            imgs = get_cells(tmp, args.mpp, args.cellsize)
            # Convert to mosaic of images
            imgs = list_to_mosaic(imgs)
            # Retrieve ax
            ax = plt.subplot(gs[i])
            # Plot mosaic
            if args.alpha:
                res = ax.imshow(imgs, interpolation='none',alpha=row.density)
            else:
                res = ax.imshow(imgs, interpolation='none')
            # Deal wit hue
            if args.hue:
                #ax.spines["top"].set_color("orange")
                ax.spines["bottom"].set_color("orange")
                ax.spines["bottom"].set_linewidth(2)
                #ax.spines["left"].set_color("orange")
                ax.spines["right"].set_color("orange")
                ax.spines["right"].set_linewidth(2)
            
            if args.text:
                if args.text == 'prob':
                    ax.text(0, 0, '{:.2f}'.format(tmp['prob'].mean()), fontsize = 10,
                            horizontalalignment='left', verticalalignment='top',
                            bbox={'facecolor':'white', 'pad':2})
            
            if args.showcoords:
                ax.text(0, 0, '{:.2f}, {:.2f}'.format(row.xo, row.yo), fontsize = 10,
                        horizontalalignment='left', verticalalignment='top',
                        bbox={'facecolor':'white', 'pad':2})
                ax.text(args.cellsize * int(np.sqrt(args.ncell)), args.cellsize * int(np.sqrt(args.ncell)), '{:.2f}, {:.2f}'.format(row.xf, row.yf), fontsize = 10,
                        horizontalalignment='right', verticalalignment='bottom',
                        bbox={'facecolor':'white', 'pad':2})
            
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
    
    if args.output:
        plt.savefig(args.output, bbox_inches='tight')
    else:
        pdb.set_trace()

if __name__ == '__main__':
    main()

#python /sc/arion/projects/comppath_liver/cellMIL/code/CMIL1_gridplot.py --data umap-local_mccv-9_rep-1.csv --ncol 10 --nrow 10 --proj1 UMAP1 --proj2 UMAP2 --ncell 25