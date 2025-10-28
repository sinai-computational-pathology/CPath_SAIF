import pandas as pd
import numpy as np
import openslide
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
import seaborn as sns

organ = 'skin'
encoder = 'SP22M'
exp_version = 'exp3'
meta_source_path = f'/sc/arion/projects/comppath_SAIF/slide_experiments/{organ}/{encoder}/{exp_version}/marimo_metadata.csv'
meta_df = pd.read_csv(meta_source_path)

# Filter DataFrame
demographic_groups = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']
meta_df = meta_df[meta_df['is_epithelium'] == 1]

# Create a boolean mask for rows where any attention score is in the top 90th percentile
mask = np.zeros(len(meta_df), dtype=bool)
for group in demographic_groups:
    attn_col = f'attn_{group}'
    threshold = meta_df[attn_col].quantile(0.90)
    mask |= (meta_df[attn_col] >= threshold)

# Filter the DataFrame using the mask
meta_df = meta_df[mask]

# Randomly select 1000 tiles per slide per race
meta_df = meta_df.groupby(['slide', 'race_curated']).apply(lambda x: x.sample(min(50, len(x)), random_state=42)).reset_index(drop=True)

# Ensure 50% correct and 50% incorrect within each slide
meta_df['correct'] = meta_df.apply(lambda row: row['race_curated'] == row['pred'], axis=1)
meta_df = meta_df.groupby(['slide', 'race_curated']).apply(lambda x: x.sample(frac=1, random_state=42)).reset_index(drop=True)
meta_df = meta_df.groupby(['slide', 'race_curated']).apply(lambda x: pd.concat([x[x['correct']].head(len(x)//2), x[~x['correct']].head(len(x)//2)])).reset_index(drop=True)

# Sample the DataFrame for plotting
sampled_meta_df = pd.concat([
    meta_df[meta_df['race_curated'] == group].sample(min(25, len(meta_df)), random_state=42)
    for group in demographic_groups
]).reset_index(drop=True)

# # Preload slides with progress bar
# slides_dict = {}
# for _, row in tqdm(sampled_meta_df.iterrows(), total=len(sampled_meta_df), desc="Preloading slides"):
#     slides_dict[row['slide']] = openslide.OpenSlide(row['dataark_path'])

plot_save_path = "/sc/arion/projects/comppath_SAIF/slide_experiments/skin/SP22M/exp3/plots/tiles_attention_plots/"
os.makedirs(plot_save_path, exist_ok=True)
    
# Function to visualize tiles
def show_slide_images(slides_dict, df):
    """
    Display patches from selected slide images categorized by race.

    Parameters:
        slides_dict (dict): Dictionary of preloaded slides.
        df (pd.DataFrame): DataFrame containing tile information.

    Returns:
        None
    """
    demographic_groups = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']
    tile_size = 224  # Default tile size

    # Grid settings: 5 rows, 5 columns for each race
    rows, cols = 5, 5

    for group in tqdm(demographic_groups, desc="Generating plots for each race"):
        fig, ax = plt.subplots(figsize=(cols * 2.5, rows * 2.5))
        group_df = df[df['race_curated'] == group].reset_index(drop=True)
        
        for idx in tqdm(range(len(group_df)), desc=f"Processing tiles for {group}", leave=False):
            row = group_df.iloc[idx]
            # Fetch the preloaded slide
            slide = slides_dict.get(row['slide'])
            if slide is None:
                raise ValueError(f"Slide {row['slide']} not found in preloaded slides.")

            img_array = np.array(slide.read_region(
                location=(int(np.round(row['x'])), int(np.round(row['y']))),
                size=(tile_size, tile_size),
                level=1
            ))
            img = Image.fromarray(img_array)
            ax.imshow(img, extent=(idx % cols, (idx % cols) + 1, idx // cols, (idx // cols) + 1))

        ax.set_xticks(np.arange(cols + 1))
        ax.set_yticks(np.arange(rows + 1))
        ax.set_xticklabels(np.arange(cols + 1))
        ax.set_yticklabels(np.arange(rows + 1))
        ax.set_title(f"Race: {group}")

        plt.subplots_adjust(hspace=0.5)
        fig.savefig(os.path.join(plot_save_path, f"tiles_attention_{group.replace('/','|')}.png"))
        plt.close(fig)

# # Generate the plots with filtered meta_df
# show_slide_images(slides_dict, sampled_meta_df)

# Run KMeans clustering on the UMAP dimensions
kmeans = KMeans(n_clusters=5, random_state=42)
meta_df['cluster'] = kmeans.fit_predict(meta_df[['UMAP_D1', 'UMAP_D2']])

# Read UMAP columns directly from meta_df
umap_d1 = meta_df['UMAP_D1']
umap_d2 = meta_df['UMAP_D2']

# Plot the UMAP results with race groups as discrete colors using seaborn
plt.figure(figsize=(10, 8))
sns.scatterplot(x='UMAP_D1', y='UMAP_D2', hue='race_curated', data=meta_df, palette='tab10', hue_order=demographic_groups, alpha=0.3)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP projection of attention scores')
plt.legend(title='Race Group')
plt.savefig(os.path.join(plot_save_path, "umap_clusters.png"))
plt.close()
