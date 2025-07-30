import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import marimo as mo
    import pandas as pd
    import altair as alt
    # alt.data_transformers.enable("vegafusion")
    import seaborn as sns
    from PIL import Image
    from pathlib import Path
    import os
    import joblib
    from sklearn.decomposition import PCA
    from cucim import CuImage
    import time
    from datetime import datetime
    from sklearn.cluster import KMeans
    import csv
    from SinaiData import pathology
    return (
        CuImage,
        Image,
        KMeans,
        PCA,
        Path,
        alt,
        csv,
        datetime,
        joblib,
        mo,
        np,
        os,
        pathology,
        pd,
        plt,
        random,
        sns,
        time,
    )


@app.cell
def __(mo):
    mo.md(
        f"""
        # **[SAIF-Skin] Histomorphological Phenotype Learning**

        **This is Histomorphological Phenotype Cluster (HPC) and Tile Vector Representation Explorer for Sinai AI Fairness (SAIF) Project**
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## Dataset Browser""")
    return


@app.cell
def __(Path):
    base_dir = Path('Emb_Viz/model_SP22M/')
    return (base_dir,)


@app.cell
def __(base_dir, mo):
    file_browser = mo.ui.file_browser(
        initial_path=base_dir,
        filetypes=['.csv'],
        multiple=False,
        restrict_navigation=False,
        selection_mode='file')

    file_browser
    return (file_browser,)


@app.cell
def __(mo):
    mo.md("""### Loading dataframe from selected metadata file""")
    return


@app.cell
def __(file_browser, os, pd):
    meta_source_path = file_browser.path(0)

    if meta_source_path is None or not os.path.exists(meta_source_path):
        print("No valid file selected or file does not exist. Using default file from marimo_metadata.csv")
        meta_source_path = '/Emb_Viz/model_SP22M/marimo_metadata.csv'

    meta_df = pd.read_csv(meta_source_path)
    meta_df['kmeans_cluster'] = meta_df['kmeans_cluster'].astype(str)
    # meta_df['slidepath'] = meta_df['slidepath'].apply(
    #     lambda x: x.replace("/data/", "/data/skin/")
    # )
    meta_df
    return meta_df, meta_source_path


@app.cell
def __(mo):
    mo.md(r"""### Selected subsamples for visualizations""")
    return


@app.cell
def __(mo):
    sampling_options = [
        "Use all data",
        # "Randomly sample a certain number of tiles",
        "Randomly sample a certain number of slides",
        "Select certain group of slides",
        "Select group of slides from .json"
    ]

    sampling_dropdown = mo.ui.dropdown(
        options=sampling_options,
        value=sampling_options[1], 
        label="Choose sampling method"
    )

    sampling_dropdown
    return sampling_dropdown, sampling_options


@app.cell
def __(mo, sampling_dropdown):
    # Text input for specifying the number of tiles to sample
    tile_count_input = mo.ui.text(
        value="1000",
        placeholder="Enter number of tiles",
        kind="text",
        label="Number of tiles per slide to sample",
        debounce=True,
    )

    # Text input for specifying the number of slides to sample
    slide_count_input = mo.ui.text(
        value="5",
        placeholder="Enter number of slides",
        kind="text",
        label="Number of slides to sample",
        debounce=True,
    )

    # Text input for specifying a list of slide names
    specific_slides_input = mo.ui.text(
        value="",
        placeholder="Enter slide names separated by commas",
        kind="text",
        label="List of slide names",
        debounce=True,
        full_width=True
    )

    # Function to return UI elements based on selected sampling method
    def get_sampling_ui():
        sampling_method = sampling_dropdown.value

        if sampling_method == "Use all data":
            return None  # No additional input needed

        # elif sampling_method == "Randomly sample a certain number of tiles":
        #     return tile_count_input

        elif sampling_method == "Randomly sample a certain number of slides":
            return slide_count_input, tile_count_input

        elif sampling_method == "Select certain group of slides":
            return specific_slides_input

        else:
            print("Invalid sampling method selected.")
            return None

    get_sampling_value = get_sampling_ui()
    get_sampling_value
    return (
        get_sampling_ui,
        get_sampling_value,
        slide_count_input,
        specific_slides_input,
        tile_count_input,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Follow these steps to use the data sampling functionality:

        + Fixed Random Seed (Reproducible Sampling):
        	1.	Toggle the Random Seed Switch ON.
        	2.	Enter a valid integer in the Random Seed Input field.
        	3.	**Action**: This seed will be used to generate a consistent subsample every time.

         + Generate a New Subsample (Resampling):
            1.  Click the Resample Button to generate a new random seed.
        	2.	**Action**: The newly generated seed will be used to produce a fresh subsample.

         + Random Seed (Non-Reproducible Sampling):
            1.	Leave the Random Seed Switch OFF.
        	2.	**Action**: The system will automatically generate a random seed for sampling.
        """
    )
    return


@app.cell
def __(mo):
    random_seed_switch = mo.ui.switch(
        value=True,
        label="Use Fixed Random Seed"
    )
    random_seed_switch
    return (random_seed_switch,)


@app.cell
def __(mo, random_seed_switch):
    if random_seed_switch.value:
        random_seed_input = mo.ui.text(
            value="42",
            label="Random Seed (if enabled)"
        )

    else:
        random_seed_input = None


    random_seed_input
    return (random_seed_input,)


@app.cell
def __(mo, random, used_seeds):
    used_seeds = set()

    def generate_new_seed(input_seed):
        """
        Generate a new random seed based on an input seed, avoiding repetition.

        Args:
            input_seed (int): The input seed to base the new seed on.

        Returns:
            int: A new, unique random seed.
        """
        global used_seeds

        # Set the random seed for reproducibility
        random.seed(input_seed)

        # Generate a new random seed and ensure it's not in the used_seeds set
        new_seed = random.randint(0, 1e6)
        while new_seed in used_seeds:
            new_seed = random.randint(0, 1e6)
        print(f"new seed: {new_seed}")

        # Add the new seed to the used_seeds set
        used_seeds.add(new_seed)
        return new_seed

    # Button to trigger re-sampling
    resample_button = mo.ui.button(
        value=None,
        on_click=lambda value: generate_new_seed(value),
        label="Resampling",
        kind="danger"
    )
    resample_button
    return generate_new_seed, resample_button, used_seeds


@app.cell
def __(
    filter_data,
    meta_df,
    random,
    random_seed_input,
    random_seed_switch,
    resample_button,
):
    if (random_seed_switch.value) and (resample_button.value is None):  # Fixed random seed is enabled
        # Use the fixed input random seed
        try:
            seed_for_sampling = int(random_seed_input.value)
            data = filter_data(meta_df, random_seed=seed_for_sampling)
            print(f"Using fixed random seed: {seed_for_sampling} for sampling")
        except ValueError:
            print("Invalid fixed random seed input. Please enter a valid integer.")
            data = None
    elif resample_button.value is not None:
        seed_for_sampling = resample_button.value
        data = filter_data(meta_df, random_seed=seed_for_sampling)
        print(f"Resampling with new random seed: {seed_for_sampling}")
    else:  # Randomly generate a seed and resample
        seed_for_sampling = random.randint(0, 1e6)
        data = filter_data(meta_df, random_seed=seed_for_sampling)
        print(f"No seeds set, Now use a randomly generated seed: {seed_for_sampling}")

    data
    return data, seed_for_sampling


@app.cell
def __(
    e,
    get_sampling_value,
    random,
    sampling_dropdown,
    specific_slides_input,
    tile_count_input,
):
    def filter_data(meta_df, random_seed=42):
        sampling_method = sampling_dropdown.value
        if sampling_method == "Use all data":
            return meta_df

        elif sampling_method == "Randomly sample a certain number of tiles":
            try:
                num_tiles = int(tile_count_input.value)
                return meta_df.sample(num_tiles, random_state=random_seed)
            except ValueError:
                print("Please enter a valid number of tiles.")
                return None

        elif sampling_method == "Randomly sample a certain number of slides":
            try:
                num_slides = int(get_sampling_value[0].value)
                num_tiles = int(get_sampling_value[1].value)

                unique_slides = meta_df["slide"].unique()
                random.seed(random_seed)
                sampled_slides = random.sample(list(unique_slides), k=num_slides)
                filtered_df = meta_df[meta_df["slide"].isin(sampled_slides)]

                # Sample tiles within each selected slide
                sampled_tiles = (
                    filtered_df.groupby("slide", group_keys=False)
                    .apply(lambda x: x.sample(n=min(num_tiles, len(x)), random_state=random_seed, replace=False))
                )

                return sampled_tiles
            except e:
                print(e)
                print("Please enter valid numbers for slides and tiles.")
                return None

        elif sampling_method == "Select certain group of slides":
            try:
                slide_list = specific_slides_input.value.split(",")
                slide_list = [s.strip() for s in slide_list]  # Remove whitespace
                mask = meta_df["slide"].str.contains('|'.join(slide_list), na=False)
                return meta_df[mask]
            except Exception as e:
                print(f"Error selecting slides: {e}")
                return None

        else:
            print("Invalid sampling method selected.")
            return None
    return (filter_data,)


@app.cell
def __(mo):
    mo.md("""### Choose additional annotation to use (Optional)""")
    return


@app.cell
def __(mo):
    annotation_file_browser = mo.ui.file_browser(
        initial_path='./',
        filetypes=['.csv'],
        multiple=True,
        restrict_navigation=False,
        selection_mode='file')

    annotation_file_browser
    return (annotation_file_browser,)


@app.cell
def __(mo):
    mo.md(r"""### Add A list of Interesting Slides as Landmark""")
    return


@app.cell
def __(Path, joblib, os, pathology, pd):
    def process_add_annotations(annotation_file_browser):
        """
        Process and visualize annotations based on the input files, updating slidepath with minerva_path.

        Args:
            annotation_file_browser (list-like): List-like object with a `.path(index)` method to retrieve file paths.

        Returns:
            pd.DataFrame: Processed DataFrame for visualization with updated slidepaths.
        """
        base_dir = Path('Emb_Viz/model_SP22M')  # Adjust if base_dir is defined elsewhere
        kmeans_model_path = base_dir / 'kmeans_model.pkl'
        source_meta_df = pd.read_csv('cohort_08_15_2024/slide_master_skin_dataark.csv')
        
        # Load the DataFrame with new slidepaths
        new_slide_df = pathology.data_ark_paths()

        if len(annotation_file_browser.value) == 0:
            raise ValueError("No annotation files provided.")

        all_cases = []

        for i in range(len(annotation_file_browser.value)):
            file_path = annotation_file_browser.path(i)
            folder_name = os.path.basename(os.path.dirname(file_path))

            if folder_name == "UMAP_samples":
                umap_data = pd.read_csv(file_path)
                umap_data['source'] = 'UMAP'
                umap_data['file_path'] = file_path

                # Use annotation as label if both are present; otherwise, merge
                if 'annotation' in umap_data.columns:
                    umap_data['label'] = umap_data['annotation']
                    umap_data.drop(columns=['annotation'], inplace=True)
                elif 'label' in umap_data.columns:
                    umap_data['label'] = umap_data['label'].fillna('').str.lower()

                all_cases.append(umap_data)

            elif folder_name == "Viewer_samples":
                if source_meta_df is None or kmeans_model_path is None:
                    raise ValueError("meta_df and kmeans_model_path must be provided for Viewer_samples.")

                # Extract metadata for the specific slide
                slide_id = os.path.splitext(os.path.basename(file_path))[0].split('_')[0]
                slide_metadata = source_meta_df[source_meta_df['slidepath'].str.contains(slide_id)].iloc[0]

                viewer_data = pd.read_csv(file_path)

                viewer_data['race_curated'] = slide_metadata['race_curated']
                viewer_data['slide'] = slide_id
                viewer_data['slidepath'] = slide_metadata['slidepath']
                viewer_data['tensor_path'] = f'./{slide_id}.pth'
                viewer_data['x'] = viewer_data['x'].round().astype(int)
                viewer_data['y'] = viewer_data['y'].round().astype(int)

                # Use annotation as label if both are present; otherwise, merge
                if 'annotation' in viewer_data.columns:
                    viewer_data['label'] = viewer_data['annotation']
                    viewer_data.drop(columns=['annotation'], inplace=True)
                elif 'label' in viewer_data.columns:
                    viewer_data['label'] = viewer_data['label'].fillna('').str.lower()

                # Cluster embedding using KMeans
                kmeans_model = joblib.load(kmeans_model_path)
                embeddings = viewer_data[["UMAP_D1", "UMAP_D2"]].to_numpy(dtype="float32")
                cluster_labels = kmeans_model.predict(embeddings)
                viewer_data.loc[:, 'kmeans_cluster'] = cluster_labels.astype(str)
                viewer_data['source'] = 'Viewer'
                viewer_data['file_path'] = file_path

                all_cases.append(viewer_data)

            else:
                print(f"Unknown folder name '{folder_name}' for file: {file_path}")

        # Concatenate all cases into a single DataFrame
        if all_cases:
            processed_df = pd.concat(all_cases, ignore_index=True)
            
            # Ensure the 'label' column is in lowercase
            if 'label' in processed_df.columns:
                processed_df['label'] = processed_df['label'].str.lower()

            # # Replace "/data/" with "/data/skin/" (original adjustment)
            processed_df['slidepath'] = processed_df['slidepath'].apply(lambda x: x.replace("/data/", "/data/skin/"))
            
            # Update slidepath with minerva_path
            processed_df['slide_base'] = processed_df['slide'].str.split('_', n=1).str[0]
            merged_df = processed_df.merge(new_slide_df[['filename', 'minerva_path']], 
                                           left_on='slide_base', 
                                           right_on='filename', 
                                           how='left')
            processed_df['slidepath'] = merged_df['minerva_path'].fillna(merged_df['slidepath'])  # Keep original if no match
            processed_df = processed_df.drop(columns=['slide_base'])  # Clean up

            return processed_df
        else:
            raise ValueError("No valid data processed.")
    return (process_add_annotations,)


@app.cell
def __(mo):
    mo.md("""### Preload Slides for Quick Visualization""")
    return


@app.cell
def __(mo):
    load_slide_button = mo.ui.run_button(label='click to preload slides')
    load_slide_button
    return (load_slide_button,)


@app.cell
def __(
    CuImage,
    annotation_file_browser,
    data,
    load_slide_button,
    mo,
    pd,
    process_add_annotations,
):
    def preload_slides(df, additional_cases=None, slides_dict=None):
        """
        Preload slides efficiently, loading only new slides that are not already in slides_dict.

        Args:
            df (pd.DataFrame): DataFrame containing 'slide' and 'slidepath' columns.
            additional_cases (pd.DataFrame, optional): DataFrame with additional cases, including the 'label' column.
            slides_dict (dict, optional): Existing dictionary of preloaded slides. Defaults to an empty dictionary if not provided.

        Returns:
            dict: Updated dictionary of loaded slides.
        """
        # Initialize slides_dict if not provided
        if slides_dict is None:
            slides_dict = {}

        if 'label' not in df.columns:
            df = df.copy()
            df.loc[:,'label'] = 'N/A'

        # Merge additional_cases into df if it exists
        if additional_cases is not None:
            df = pd.concat([df, additional_cases], ignore_index=True)
            print("Merged additional cases into df.")

        print(df.label.value_counts())

        unique_slides = df[['slide', 'slidepath']].drop_duplicates()

        # Identify slides that need to be loaded
        slides_to_load = unique_slides[~unique_slides['slide'].isin(slides_dict.keys())]

        if slides_to_load.empty:
            print("All required slides are already preloaded.")
            return slides_dict

        print(f"Loading {len(slides_to_load)} new slides...")
        with mo.status.progress_bar(
            title='Hi! Preloading slides for you now... Thanks for being patient!',
            subtitle='It will take a while depending on slides you chose.',
            total=len(slides_to_load)) as progress:
            for i, row in slides_to_load.iterrows():
                try:
                    slides_dict[row.slide] = CuImage(row.slidepath)
                except Exception as e:
                    print(f"\nError loading slide {row.slide}: {e}")
                    continue
                progress.update(1)

        print("Slide preloading complete.")
        if 'kmeans_cluster' in df.columns:
            df['kmeans_cluster'] = df['kmeans_cluster'].apply(
                lambda x: x if str(x).startswith("KMC:") else f"KMC:{x}"
            )

        columns_to_drop = ['UMAP_D1_min', 'UMAP_D1_max', 'UMAP_D2_min', 'UMAP_D2_max', 'file_path']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        return slides_dict, df

    # # Only loads new slides not already in current dictionary
    mo.stop(not load_slide_button.value)

    if len(annotation_file_browser.value)>0:
        additional_cases = process_add_annotations(annotation_file_browser)
        slides_dict, data_2_visualize = preload_slides(data, additional_cases)
    else:
        slides_dict, data_2_visualize = preload_slides(data)
    return additional_cases, data_2_visualize, preload_slides, slides_dict


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Tile Vector Visualization
        ---------------------------------------------------------------------------------------
        """
    )
    return


@app.cell
def __(data_2_visualize, mo):
    exclude_columns = ['x','y', 'PC1', 'PC2', 'UMAP_D1', 'UMAP_D2', 'tensor_path', 'slide', 'slidepath','kmeans_cluster']

    column_options = [col for col in data_2_visualize.columns if col not in exclude_columns]

    column_dropdown = mo.ui.dropdown(
      options=column_options,
      value='label',
      label='Choose annotation field for representations'
    )
    column_dropdown
    return column_dropdown, column_options, exclude_columns


@app.cell
def __(mo):
    scatter_size = mo.ui.slider(start=4, stop=50, step=2, label='Scatter size')
    scatter_size
    return (scatter_size,)


@app.cell
def __(mo):
    row_limit = mo.ui.slider(start=0, stop=10000, step=500, value=1000, label='Maxium Patches to show')
    row_limit
    return (row_limit,)


@app.cell
def __(column_dropdown, data_2_visualize):
    selection = column_dropdown.value
    data_sub  = data_2_visualize[['UMAP_D1', 'UMAP_D2', 'race_curated', 'label', 'kmeans_cluster','slide', 'x','y','source']]
    selection = selection.replace('.','p')
    return data_sub, selection


@app.cell
def __(data_2_visualize, mo, selection):
    cluster_options = data_2_visualize[selection].unique()

    multi_select_dropdown = mo.ui.multiselect(
        options=cluster_options,
        value=cluster_options,
        label=f"Select Column '{selection}' to Visualize"
    )
    multi_select_dropdown
    return cluster_options, multi_select_dropdown


@app.cell
def __(mo):
    protect_class_switch = mo.ui.switch(label='Protect Interested Class',value=True)
    protect_class_switch
    return (protect_class_switch,)


@app.cell
def __(cluster_options, mo, protect_class_switch, selection):
    if protect_class_switch.value:

        if selection == 'race_curated':
            default_value = cluster_options
        elif selection == 'label':
            default_value = [cluster_options[1]] # select epithilium
        else:
            default_value = []

        protected_classes_select = mo.ui.multiselect(
            options=cluster_options,
            value=default_value,
            label=f"Select Classes in {selection} to be Protected from downsampling"
        )
    else:
        protected_classes_select = None

    protected_classes_select
    return default_value, protected_classes_select


@app.cell
def __(
    data_sub,
    multi_select_dropdown,
    pd,
    protected_classes_select,
    row_limit,
    selection,
):
    if multi_select_dropdown.value:
        filtered_data = data_sub[data_sub[selection].isin([x for x in multi_select_dropdown.value])]
    else:
        filtered_data = data_sub

    # Check if the filtered data exceeds the row limit
    if (len(filtered_data) > row_limit.value) and (protected_classes_select is not None) and len(protected_classes_select.value) > 0:
        print(f"Rows exceend set row limits ({row_limit.value}), now downsampling... \n")
        protected_rows = filtered_data[filtered_data[selection].isin(protected_classes_select.value)]
        non_protected_rows = filtered_data[~filtered_data[selection].isin(protected_classes_select.value)]

        if len(protected_rows) < row_limit.value: # if it doesn't reach limit
            downsampled_rows = non_protected_rows.sample(
                n=max(row_limit.value - len(protected_rows), 0)
            )

            filtered_data = pd.concat([protected_rows, downsampled_rows])
        else:
            filtered_data = protected_rows

        print(f"Found {len(protected_rows)} rows in protected classes ({protected_classes_select.value}). \n")
        print(f"Filtered data downsampled to {len(filtered_data)} rows, with all protected class rows included.")

    else:
        print("Filtered data is within the row limit; no downsampling applied.")
    return (
        downsampled_rows,
        filtered_data,
        non_protected_rows,
        protected_rows,
    )


@app.cell
def __(mo):
    controls = mo.ui.radio(["Selection", "Pan & Zoom"], value="Selection")
    return (controls,)


@app.cell
def __(
    alt,
    cluster_options,
    column_dropdown,
    controls,
    filtered_data,
    mo,
    scatter_size,
    selection,
):
    # Figure 1: Scatter
    color_scale = alt.Scale(domain=cluster_options, scheme='category10')

    labeling = alt.Color(
        '%s:N' % selection,
        scale=color_scale,
        legend=alt.Legend(title=selection)
    )


    alt_chart = alt.Chart(filtered_data, width=500, height=500).mark_circle(size=scatter_size.value).encode(
    x=alt.X("UMAP_D1:Q"),
    y=alt.Y("UMAP_D2:Q"),
    color=labeling
    )

    scatter_chart = mo.ui.altair_chart(
            alt_chart.interactive() if controls.value == "Pan & Zoom" else alt_chart,
            chart_selection=controls.value == "Selection",
            legend_selection=controls.value == "Selection", 
        )

    scatter = mo.vstack([scatter_chart, controls, column_dropdown])
    return alt_chart, color_scale, labeling, scatter, scatter_chart


@app.cell
def __(
    alt,
    alt_chart,
    color_scale,
    controls,
    filtered_data,
    mo,
    pd,
    selection,
):
    # Figure 2: Scatter Plot with Faceted Marginal Histrogram
    # Define histograms
    filtered_data['UMAP_D1'] = pd.to_numeric(filtered_data['UMAP_D1'], errors='coerce').fillna(0)
    filtered_data['UMAP_D2'] = pd.to_numeric(filtered_data['UMAP_D2'], errors='coerce').fillna(0)

    xscale = alt.Scale(domain=(filtered_data["UMAP_D1"].min(), filtered_data["UMAP_D1"].max()))
    yscale = alt.Scale(domain=(filtered_data["UMAP_D2"].min(), filtered_data["UMAP_D2"].max()))

    top_hist = (
        alt.Chart(filtered_data)
        .mark_bar(opacity=0.3, binSpacing=0)
        .encode(
            alt.X("UMAP_D1:Q")
                .bin(maxbins=20, extent=xscale.domain)
                .stack(None)
                .title(""),
            alt.Y("count()")
                .stack(None)
                .title(""),
            alt.Color("%s:N" % selection, scale=color_scale),
        )
        .properties(height=100, width=800)
    )

    right_hist = (
        alt.Chart(filtered_data)
        .mark_bar(opacity=0.3, binSpacing=0)
        .encode(
            alt.Y("UMAP_D2:Q")
                .bin(maxbins=20, extent=yscale.domain)
                .stack(None)
                .title(""),
            alt.X("count()")
                .stack(None)
                .title(""),
            alt.Color("%s:N" % selection, scale=color_scale),
        )
        .properties(height=800, width=100)
    )

    combined_chart = top_hist & (alt_chart | right_hist)

    # Display in Marimo
    scatter_chart_2 = mo.ui.altair_chart(combined_chart)

    scatter_2 = mo.vstack([scatter_chart_2, controls])
    return (
        combined_chart,
        right_hist,
        scatter_2,
        scatter_chart_2,
        top_hist,
        xscale,
        yscale,
    )


@app.cell
def __(alt, combined_chart, filtered_data, mo, scatter, scatter_size):
    # Generate scatter plot tabs dynamically based on columns
    def create_scatter_tabs(filtered_data, column_list, scatter_size):
        tabs = {}

        # Fixed tab for "Interactive Mode"
        tabs["☀️ Scatter (Interactive Mode)"] = scatter

        # Loop through column list to create scatter plots
        for column in column_list:
            color_scale = alt.Scale(domain=filtered_data[column].unique(), scheme='category10')
            labeling = alt.Color(
                f"{column}:N",
                scale=color_scale,
                legend=alt.Legend(title=column.replace("_", " ").capitalize())
            )

            alt_chart = alt.Chart(
                filtered_data,
                width=800,
                height=800
            ).mark_circle(size=scatter_size.value).encode(
                x=alt.X("UMAP_D1:Q"),
                y=alt.Y("UMAP_D2:Q"),
                color=labeling
            )

            # Add to tabs
            tabs[f"⭐️ Scatter ({column.replace('_', ' ').capitalize()})"] = alt_chart

        # Fixed tab for histogram
        tabs["⭐️ Scatter (Histogram)"] = combined_chart

        return mo.ui.tabs(tabs)

    # Specify the column list to create scatter plots for
    scatter_columns = ['race_curated', 'kmeans_cluster', 'label']  # Replace with actual column names
    scatter_tabs = create_scatter_tabs(filtered_data, scatter_columns, scatter_size)
    scatter_tabs
    return create_scatter_tabs, scatter_columns, scatter_tabs


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Patches Visualization Section
        ---------------------------------------------------------------------------------------
        """
    )
    return


@app.cell
def __(mo):
    extra_context_slider = mo.ui.slider(
        start=0,
        stop=1024,
        step=16,
        label="Extra context size for tiles",
        value=64
    )

    extra_context_slider
    return (extra_context_slider,)


@app.cell
def __(mo):
    max_images_slider = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        label="Maximum number of images to display",
        value=20
    )

    max_images_slider
    return (max_images_slider,)


@app.cell
def __(mo, scatter_chart):
    table = mo.ui.table(scatter_chart.value, page_size=5)
    return (table,)


@app.cell
def __(
    data_2_visualize,
    extra_context_slider,
    max_images_slider,
    mo,
    scatter_chart,
    show_slide_images,
    slides_dict,
    table,
):
    mo.stop(not len(scatter_chart.value))

    extra_context = extra_context_slider.value
    max_NO_images = max_images_slider.value

    selected_images = (
        show_slide_images(list(scatter_chart.value.index), slides_dict, data_2_visualize, extra_context, max_NO_images)
    )
    # selected_images
    mo.md(
        f"""
        **Data Selected:**
        {mo.as_html(selected_images)}

        {table}
        """
    )
    return extra_context, max_NO_images, selected_images


@app.cell
def __(Image, np, os, plt):
    def update_tile_coordinates(df, new_tile_size, tile_size=224):
        offset = (new_tile_size - tile_size) // 2
        df = df.copy()
        df['x'] = df['x'] - offset
        df['y'] = df['y'] - offset
        return df

    def show_slide_images(indices, slides_dict, df, extra_context, max_NO_images=20):
        tile_size = 224 # default tile_size
        new_tile_size = tile_size + extra_context
        selected_rows = df.iloc[indices]

        if len(selected_rows) > max_NO_images:
            selected_rows = selected_rows.sample(max_NO_images, random_state=42)

        selected_indices = selected_rows.index.tolist()
        selected_rows = selected_rows.copy()
        selected_rows = selected_rows[selected_rows['slidepath'].apply(os.path.isfile)]
        selected_rows = update_tile_coordinates(selected_rows, new_tile_size, tile_size)
        num_images = len(selected_rows)

        if num_images == 0:
            raise ValueError("No valid slide paths found in the selected indices.")

        # Determine grid size based on the number of images
        rows = (num_images + 5) // 6  # 6 images per row
        fig, axes = plt.subplots(rows, 6, figsize=(12, 2 * rows))
        axes = axes.flatten()

        for idx, (ax, (_, row)) in enumerate(zip(axes, selected_rows.iterrows())):
            try:
                # Fetch the preloaded slide
                slide = slides_dict.get(row['slide'])
                if slide is None:
                    raise ValueError(f"Slide {row['slide']} not found in preloaded slides.")

                img_array = np.array(slide.read_region(location=(int(np.round(row['x'])), int(np.round(row['y']))), size=(new_tile_size, new_tile_size), level=1))
                img = Image.fromarray(img_array)
                ax.imshow(img)
            except Exception as e:
                print(e)
                ax.text(0.5, 0.5, "NA sry", ha='center', va='center', fontsize=8)

            original_index = selected_indices[idx]
            ax.set_title(f"{original_index}", fontsize=12, fontweight='bold')
            ax.set_yticks([])
            ax.set_xticks([])

        # Hide any unused axes
        for ax in axes[idx + 1:]:
            ax.axis('off')

        plt.tight_layout()
        return fig
    return show_slide_images, update_tile_coordinates


@app.cell
def __(mo):
    mo.md("""## Leave Comments on Selected Cases in UMAP""")
    return


@app.cell
def __(mo):
    working_user_input = mo.ui.text(
        value="",
        placeholder="Enter your full name for annotation records",
        label="Username",
        full_width=True
    )

    working_user_input
    return (working_user_input,)


@app.cell
def __(mo):
    group_annotation_input = mo.ui.text(
        value="",
        placeholder="Enter group annotation for selected points",
        label="Group Annotation",
        full_width=True
    )

    group_annotation_input
    return (group_annotation_input,)


@app.cell
def __(mo):
    individual_annotation_input = mo.ui.text_area(
        value="",
        placeholder="Enter individual annotation for selected point",
        label="Individual Annotation",
        full_width=True
    )

    individual_annotation_input
    return (individual_annotation_input,)


@app.cell
def __(mo):
    save_annotation_button = mo.ui.run_button(
        label="Save Annotations to csv"
    )
    save_annotation_button
    return (save_annotation_button,)


@app.cell
def __(
    csv,
    data_2_visualize,
    datetime,
    group_annotation_input,
    individual_annotation_input,
    mo,
    os,
    pd,
    save_annotation_button,
    scatter_chart,
    table,
    working_user_input,
):
    annotated_df = pd.DataFrame()

    def save_annotations_to_csv(filename, data, selected_indices, annotation, chart_selections, is_group):
        # Prepare data for saving
        selected_rows = data.loc[selected_indices].copy()
        selected_rows["annotation"] = annotation

        interval_columns = []
        for key, intervals in chart_selections.get("select_interval", {}).items():
            selected_rows[f"{key}_min"] = intervals[0]
            selected_rows[f"{key}_max"] = intervals[1]
            interval_columns.extend([f"{key}_min", f"{key}_max"])

        # Check if file exists and append
        file_exists = os.path.isfile(filename)
        with open(os.path.join('Emb_Viz/','UMAP_samples',filename), mode="a", newline="") as file:
            writer = csv.writer(file)
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow([*data.columns, "annotation", *interval_columns])
            # Append data
            writer.writerows(selected_rows.reset_index(drop=True).values)

        # Notification output
        annotation_type = "Group" if is_group else "Individual"
        mo.md(
            f"""
            **{annotation_type} annotation saved.**
            File updated: {filename}
            """
        )

    def save_annotations_to_new_df(meta_df, selected_indices, annotation, chart_selections, filename, is_group):
        global annotated_df

        # Save selected rows to a new DataFrame
        selected_rows = meta_df.loc[selected_indices].copy()
        selected_rows["annotation"] = annotation

        for key, intervals in chart_selections.get("select_interval", {}).items():
            selected_rows[f"{key}_min"] = intervals[0]
            selected_rows[f"{key}_max"] = intervals[1]

        # Update the global DataFrame
        annotated_df = pd.concat([annotated_df, selected_rows], ignore_index=True)

        # Save to CSV
        save_annotations_to_csv(filename, meta_df, selected_indices, annotation, chart_selections, is_group)


    mo.stop(not save_annotation_button.value)

    user_input = working_user_input.value
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{user_input}_{date_str}.csv"

    # Save Group Annotation
    if len(group_annotation_input.value) > 0:
        selected_indices = scatter_chart.value.index  # Selected scatter points
        annotation = group_annotation_input.value
        chart_selections = scatter_chart.selections  # Coordination of bounding boxes
        save_annotations_to_new_df(data_2_visualize, selected_indices, annotation, chart_selections, filename, is_group=True)

    # Save Individual Annotation
    if len(individual_annotation_input.value) > 0:
        indivisual_selected_indices = table.value.index  # Selected rows from table
        individual_annotation = individual_annotation_input.value
        indivisual_chart_selections = table.value
        save_annotations_to_new_df(
            data_2_visualize,
            indivisual_selected_indices,
            individual_annotation,
            indivisual_chart_selections,
            filename,
            is_group=False,
        )

    mo.md(
        f"""
        **Annotated rows saved to:**
        {filename}

        {mo.ui.table(annotated_df, page_size=10)}
        """
    )
    return (
        annotated_df,
        annotation,
        chart_selections,
        date_str,
        filename,
        individual_annotation,
        indivisual_chart_selections,
        indivisual_selected_indices,
        save_annotations_to_csv,
        save_annotations_to_new_df,
        selected_indices,
        user_input,
    )


@app.cell
def __(mo):
    mo.md("""## Classifier!""")
    return


@app.cell
def __():
    import openslide
    from matplotlib.patches import Rectangle
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from matplotlib.colors import Normalize
    from SlideTileExtractor import extract_tissue
    return (
        LogisticRegression,
        Normalize,
        Rectangle,
        accuracy_score,
        classification_report,
        extract_tissue,
        openslide,
        train_test_split,
    )


@app.cell
def __(
    LogisticRegression,
    accuracy_score,
    classification_report,
    datetime,
    joblib,
    pd,
    train_test_split,
):
    def classify_epithelium(df, method="cluster", test_size=0.4, random_state=42):
        if method == "cluster":
            # Simple rule-based classification based on KMeans cluster
            df['is_epithelium'] = df['kmeans_cluster'].apply(lambda x: str(x) == "2")
            model = None

        elif method == "logistic":
            # Filter labeled patches
            labeled_patches = df[df['label'].notna()]
            positive_samples = labeled_patches[labeled_patches['label'].str.lower() == "epithelium"]

            # Select negative samples (unlabeled patches)
            negative_samples = df[df['label']=='non-epithelium'].sample(
                n=len(positive_samples) * 1, random_state=random_state
            )

            # Combine positive and negative samples
            training_data = pd.concat([positive_samples, negative_samples])
            X = training_data[['UMAP_D1', 'UMAP_D2']].values
            y = (training_data['label'].str.lower() == "epithelium").fillna(False).astype(int)

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Train logistic regression model
            model = LogisticRegression(random_state=random_state)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {test_accuracy:.2f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred, target_names=["Non-Epithelium", "Epithelium"]))

            # Classify all patches using the model
            X_all = df[['UMAP_D1', 'UMAP_D2']].values
            df['is_epithelium'] = model.predict(X_all)

            # Save the trained model
            today_date = datetime.now().strftime("%Y-%m-%d")
            model_filename = f"model_{method}_{today_date}.joblib"
            joblib.dump(model, model_filename)
            print(f"Model saved as {model_filename}")

        else:
            raise ValueError("Invalid method. Choose 'cluster' or 'logistic'.")

        return df, model
    return (classify_epithelium,)


@app.cell
def __(classify_epithelium, data_2_visualize):
    data_3_visualize, trained_model = classify_epithelium(data_2_visualize, method="logistic")
    return data_3_visualize, trained_model


@app.cell
def __(extract_tissue, np, openslide, os, pd, plt):
    import matplotlib.patches as patches

    def extract_slide_name(file_name):
        """
        Extract the substring before the second underscore.

        Args:
            file_name (str): The input string.

        Returns:
            str: The substring before the second underscore.
        """
        parts = file_name.split('_')
        return '_'.join(parts[:2])

    def find_and_merge_csv(folder_path, slidename):
        """
        Find and merge 5 specific .csv files related to a given slidename into a single DataFrame.

        Args:
            folder_path (str): Path to the folder containing .csv files.
            slidename (str): The unique identifier for the slide (e.g., from extract_before_second_underscore).

        Returns:
            pd.DataFrame: A DataFrame with separate columns for each attention type (Asian, White, etc.).
        """
        # File patterns to search for
        suffix_mapping = {
            "_A.csv": "Asian",
            "_W.csv": "White",
            "_B.csv": "Black",
            "_H.csv": "Hispanic/Latino",
            "_O.csv": "Other"
        }

        # Initialize an empty list to hold individual DataFrames
        dataframes = []

        for suffix, label in suffix_mapping.items():
            filename = f"{slidename}{suffix}"
            file_path = os.path.join(folder_path, filename)

            if os.path.exists(file_path):
                # Read the CSV file
                df = pd.read_csv(file_path, header=None, names=['x', 'y', 'race_attn'])
                # Add a column for the attention type
                df[label] = df['race_attn']
                df['slide'] = slidename  # Add slide identifier
                df = df.drop(columns=['race_attn'])  # Drop the original 'race_attn' column
                dataframes.append(df)
            else:
                print(f"File not found: {file_path}")

        # Merge all DataFrames on 'x', 'y', and 'slide' columns
        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
            merged_df = merged_df.groupby(['x', 'y', 'slide'], as_index=False).first()  # Avoid duplicates
        else:
            merged_df = pd.DataFrame()

        return merged_df

    def attention_plot(slide_path, df, model, attn_column, resolution_level, patch_size=448):
        """
        Visualize the attention map with a grid overlay on a thumbnail, allowing for dynamic alpha changes.

        Args:
            slide_path (str): Path to the whole slide image.
            df (pandas.DataFrame): DataFrame with 'x', 'y', and attention values.
            model: Machine learning model to predict attention scores.
            attn_column (str): Column in df for attention values.
            resolution_level (int): Desired OpenSlide resolution level for visualization.
            patch_size (int): Patch size in pixels at the original resolution.
            alpha (float): Transparency level for the patches.

        Returns:
            tuple: Matplotlib figure and axis objects for further manipulation.
        """
        # Open the slide
        slide = openslide.OpenSlide(slide_path)

        # Get dimensions for the desired resolution level
        thumb_w, thumb_h = slide.level_dimensions[resolution_level]

        # Read the full image at the desired resolution level
        thumbnail = slide.read_region((0, 0), resolution_level, (thumb_w, thumb_h)).convert("RGB")

        ref_level, ref_mult = extract_tissue.find_level(slide, 1.0, patchsize=patch_size, base_mpp=None)

        # Filter data for the specific slide
        # slide_df = df[df['slidepath'] == slide_path].copy()
        X_all = df[['UMAP_D1', 'UMAP_D2']].values

        if attn_column == 'is_epithelium':
            if not model is None:
                df[attn_column] = model.predict(X_all)
            else:
                df['is_epithelium'] = df['kmeans_cluster'].apply(lambda x: str(x) == "2")

        x_coords = df['x'].values
        y_coords = df['y'].values
        attn_values = df[attn_column].values

        # Scale coordinates and patch size for the desired resolution level
        scale_factor = slide.level_downsamples[resolution_level]
        scaled_x = (np.round(x_coords / scale_factor)).astype(int)
        scaled_y = (np.round(y_coords / scale_factor)).astype(int)
        scaled_patch_size = int(np.round(patch_size / scale_factor))

        # Plot thumbnail
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(thumbnail, alpha=1.0)

        # Store rectangles for dynamic alpha changes
        rectangles = []

        # Draw grid and mark attention patches
        for x, y, value in zip(scaled_x, scaled_y, attn_values):
            # Define the region corresponding to this patch
            x_start = x
            y_start = y

            # Add a rectangle for the patch
            rect = patches.Rectangle(
                (x_start, y_start),  # Bottom-left corner
                scaled_patch_size,  # Width
                scaled_patch_size,  # Height
                facecolor="red" if value == 1 else "blue",
                alpha=0.5
            )
            rectangles.append(rect)  # Store the rectangle for later manipulation
            ax.add_patch(rect)

        # Adjust visualization settings
        ax.set_xlim(0, thumb_w)
        ax.set_ylim(thumb_h, 0)  # Invert y-axis for correct orientation
        ax.axis("off")
        plt.title(f"Attention Plot")

        return fig, ax, rectangles
    return attention_plot, extract_slide_name, find_and_merge_csv, patches


@app.cell
def __(mo):
    exp_options = ['exp1','exp2','exp3','exp4']

    exp_dropdown = mo.ui.dropdown(
      options=exp_options,
      value='exp1',
      label='Choose skin cohort experiment version to visualize'
    )

    exp_dropdown
    return exp_dropdown, exp_options


@app.cell
def __(exp_dropdown, mo):
    attention_file_browser = mo.ui.file_browser(
        initial_path=f'slide_experiments/skin/{exp_dropdown.value}',
        multiple=True,
        restrict_navigation=False,
        selection_mode='file')

    attention_file_browser
    return (attention_file_browser,)


@app.cell
def __(mo):
    mult_slider = mo.ui.slider(value=5, start=3, stop=9, step=1, label='Mult')
    mult_slider
    return (mult_slider,)


@app.cell
def __(mo):
    attn_options = ['is_epithelium','White','Black','Asian','Other','Hispanic/Latino']

    attn_dropdown = mo.ui.dropdown(
      options=attn_options,
      value='is_epithelium',
      label='Choose attention to visualize'
    )

    attn_dropdown
    return attn_dropdown, attn_options


@app.cell
def __(
    Path,
    attention_file_browser,
    extract_slide_name,
    find_and_merge_csv,
    meta_df,
    pd,
):
    slidename = extract_slide_name(attention_file_browser.name(0))
    folder_path = Path(attention_file_browser.path(0)).parents[0]

    slide_path = meta_df[meta_df['slide']==slidename]['slidepath'].iloc[0]
    # # sub slide_df
    slide_df = meta_df[meta_df['slide'] == slidename].copy()
    # sub attn_df
    attn_df = find_and_merge_csv(folder_path, slidename)
    merge_df = pd.merge(slide_df, attn_df, on=['x', 'y', 'slide'], how='inner')
    merge_df
    return attn_df, folder_path, merge_df, slide_df, slide_path, slidename


@app.cell
def __(
    attention_plot,
    attn_dropdown,
    merge_df,
    mult_slider,
    slide_path,
    trained_model,
):
    fig, ax, rectangles = attention_plot(slide_path, merge_df, trained_model, attn_column=attn_dropdown.value, resolution_level=mult_slider.value)
    return ax, fig, rectangles


@app.cell
def __(mo):
    alpha_slider = mo.ui.slider(value=0.5, start=0, stop=1, step=0.1, label='alpha')
    alpha_slider
    return (alpha_slider,)


@app.cell
def __(alpha_slider, fig, rectangles):
    def update_alpha(rectangles, new_alpha):
        """
        Update the alpha transparency of all rectangles.

        Args:
            rectangles (list): List of matplotlib.patches.Rectangle objects.
            new_alpha (float): New alpha value to apply.

        Returns:
            None
        """
        for rect in rectangles:
            rect.set_alpha(new_alpha)

    update_alpha(rectangles, new_alpha=alpha_slider.value)
    fig.canvas.draw()
    fig
    return (update_alpha,)


@app.cell
def __(exp_dropdown, pd):
    # merge data in validation set and merge ICD code into it to find representative slides
    attention_version = exp_dropdown.value
    slides_df = pd.read_csv('cohort_08_15_2024/slide_master_skin_dataark.csv')
    icd_metadata = pd.read_csv('cohort_08_15_2024/metadata_skin_icd.csv')
    master_df = pd.merge(slides_df, icd_metadata[['accession_no','code','description']], on=['accession_no'],how='left')
    return attention_version, icd_metadata, master_df, slides_df


@app.cell
def __(generate_json_with_heatmaps, master_df, meta_df, trained_model):
    selected_cases = master_df[(master_df.split=='val') & (master_df.division=='DERM') &(master_df['code'].str.startswith(('C','D23')))]
    # selected_cases
    generate_json_with_heatmaps(selected_cases, exp_version='exp1', meta_df=meta_df, trained_model=trained_model, version_name='100124_SAIF_skin_exp1_Kevin')
    return (selected_cases,)


@app.cell
def __(mo):
    mo.md("""# Distribution Analysis of Attention Heatmaps""")
    return


@app.cell
def __():
    from scipy.stats import entropy, kurtosis
    return entropy, kurtosis


@app.cell
def __(mo):
    exps_options = ['exp1', 'exp2', 'exp3', 'exp4']

    mul_exps_dropdown = mo.ui.multiselect(
        options=exps_options,
        value=exps_options,
        label="Choose experiments to visualize"
    )

    mul_exps_dropdown
    return exps_options, mul_exps_dropdown


@app.cell
def __(
    Path,
    entropy,
    kurtosis,
    load_attention_scores,
    mul_exps_dropdown,
    np,
    plt,
):
    experiments = mul_exps_dropdown.value
    base_path = Path('slide_experiments/skin/')
    demographic_groups = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']

    def calculate_homogeneity_metrics(attention_scores):
        # Initialize dictionary to store homogeneity metrics for each race group
        metrics = {
            'entropy': {},
            'std_dev': {},
            'kurtosis': {}
        }

        # Calculate metrics for each race group independently
        for i, race_scores in enumerate(attention_scores):
            race_group = demographic_groups[i]

            # Ensure the scores sum to 1 by normalizing, to represent a probability distribution
            if race_scores.sum() > 0:
                race_probs = race_scores / race_scores.sum()
            else:
                # If all scores are zero, set a uniform distribution as fallback
                race_probs = np.ones_like(race_scores) / len(race_scores)

            # Clip to avoid log(0) issues in entropy calculation
            race_probs = np.clip(race_probs, 1e-10, None)

            # Calculate entropy, standard deviation, and kurtosis for the race group
            metrics['entropy'][race_group] = entropy(race_probs, base=2)
            metrics['std_dev'][race_group] = np.std(race_scores)
            metrics['kurtosis'][race_group] = kurtosis(race_scores)

        return metrics

    # Function to process all slides in an experiment and demographic group
    def process_experiment(exp_id):
        attention_scores_dict, _, _ = load_attention_scores(exp_id)
        metrics_per_slide = {}

        for slide_id, scores in attention_scores_dict.items():
            metrics = calculate_homogeneity_metrics(scores)
            metrics_per_slide[slide_id] = metrics

        return metrics_per_slide

    def aggregate_metrics(metrics_per_slide):
        # Initialize dictionaries to hold aggregated metrics for each demographic group
        aggregated_metrics = {
            'entropy': {group: [] for group in demographic_groups},
            'std_dev': {group: [] for group in demographic_groups},
            'kurtosis': {group: [] for group in demographic_groups}
        }

        # Populate the aggregated metrics for each race group
        for slide_metrics in metrics_per_slide.values():
            for group in demographic_groups:
                aggregated_metrics['entropy'][group].append(slide_metrics['entropy'][group])
                aggregated_metrics['std_dev'][group].append(slide_metrics['std_dev'][group])
                aggregated_metrics['kurtosis'][group].append(slide_metrics['kurtosis'][group])

        return aggregated_metrics

    def plot_distribution(aggregated_metrics_dict):
        # Demographic groups for labeling
        demographic_groups = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']
        experiments = list(aggregated_metrics_dict.keys())  # List of experiment IDs

        # Define colors for each experiment's histogram (customize as needed)
        colors = ['skyblue', 'salmon', 'lightgreen', 'plum']

        # Define figure with 3 rows (one for each metric) and 5 columns (one for each race group)
        fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharey=False)
        fig.suptitle('Homogeneity Metrics Distribution by Race Group Across Experiments', fontsize=16)

        # Metrics and their row indices
        metrics = ['entropy', 'std_dev', 'kurtosis']
        row_titles = ['Entropy', 'Std Dev', 'Kurtosis']

        # Calculate global x and y limits for each metric
        global_limits = {}
        for metric in metrics:
            all_values = []
            for exp_id in experiments:
                for group in demographic_groups:
                    all_values.extend(aggregated_metrics_dict[exp_id][metric][group])
            global_min_x, global_max_x = min(all_values), max(all_values)
            global_min_y, global_max_y = 0, 0  # Initialize y-axis limits

            # Estimate global y-axis limits based on density histograms
            for exp_id in experiments:
                for group in demographic_groups:
                    counts, bin_edges = np.histogram(
                        aggregated_metrics_dict[exp_id][metric][group],
                        bins=20,
                        density=True
                    )
                    global_max_y = max(global_max_y, counts.max())

            # Store limits for this metric
            global_limits[metric] = {
                'xlim': (global_min_x, global_max_x),
                'ylim': (global_min_y, global_max_y)
            }

        # Plot each metric for each demographic group
        for row_idx, metric in enumerate(metrics):
            for col_idx, group in enumerate(demographic_groups):
                # Plot histograms for all experiments
                for exp_id_idx, exp_id in enumerate(experiments):
                    axes[row_idx, col_idx].hist(
                        aggregated_metrics_dict[exp_id][metric][group],
                        bins=20,
                        color=colors[exp_id_idx],
                        alpha=0.5,
                        label=f'{exp_id}' if col_idx == 0 else "",  # Add legend only for the first column
                        edgecolor='black',
                        density=True
                    )

                # Set shared limits for the current metric
                axes[row_idx, col_idx].set_xlim(global_limits[metric]['xlim'])
                axes[row_idx, col_idx].set_ylim(global_limits[metric]['ylim'])

                # Add titles and labels
                axes[row_idx, col_idx].set_title(f'{group} - {row_titles[row_idx]}')
                axes[row_idx, col_idx].set_xlabel(row_titles[row_idx])
                axes[row_idx, col_idx].set_ylabel('Density' if col_idx == 0 else "")

        # Add legend for the first column in each row
        for row_idx, ax in enumerate(axes[:, 0]):
            ax.legend(title='Experiment')

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    return (
        aggregate_metrics,
        base_path,
        calculate_homogeneity_metrics,
        demographic_groups,
        experiments,
        plot_distribution,
        process_experiment,
    )


@app.cell
def __(
    aggregate_metrics,
    experiments,
    plot_distribution,
    process_experiment,
):
    aggregated_metrics_dict = {}

    # Process each experiment in the list
    for exp in experiments:
        print(f'Processing {exp}...')
        metrics_per_slide = process_experiment(exp)
        aggregated_metrics_dict[exp] = aggregate_metrics(metrics_per_slide)

    # Visualize the aggregated metrics
    plot_distribution(aggregated_metrics_dict)
    return aggregated_metrics_dict, exp, metrics_per_slide


@app.cell
def __(aggregated_metrics_dict, plt):
    def plot_single_metric_distribution(aggregated_metrics_dict, metric):
        # Demographic groups for labeling
        demographic_groups = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']
        experiments = list(aggregated_metrics_dict.keys())  # List of experiment IDs

        # # Calculate global x and y limits for "Std Dev" across all experiments and demographic groups
        # all_std_dev_values = []
        # for exp_id in experiments:
        #     for group in demographic_groups:
        #         all_std_dev_values.extend(aggregated_metrics_dict[exp_id][metric][group])
        # global_xlim = (min(all_std_dev_values), max(all_std_dev_values))
        # global_ylim = (0, max(
        #     np.histogram(all_std_dev_values, bins=20, density=True)[0] + 0.75
        # ))

        # Define colors for each demographic group (customize as needed)
        colors = ['skyblue', 'salmon', 'lightgreen', 'plum', 'gold']

        # Define figure with rows for experiments and columns for demographic groups
        fig, axes = plt.subplots(len(experiments), len(demographic_groups), figsize=(20, 4 * len(experiments)), sharey=True, sharex=True)
        fig.suptitle('Std Dev Distribution by Race Group Across Experiments', fontsize=16)

        # Iterate over experiments and demographic groups
        for row_idx, exp_id in enumerate(experiments):
            for col_idx, group in enumerate(demographic_groups):
                # Get data for the current experiment and demographic group
                data = aggregated_metrics_dict[exp_id][metric][group]

                # Plot histogram
                axes[row_idx, col_idx].hist(
                    data,
                    bins=20,
                    color=colors[col_idx],
                    alpha=0.7,
                    edgecolor='black',
                    density=True
                )

                # # Set shared x and y limits
                # axes[row_idx, col_idx].set_xlim(global_xlim)
                # axes[row_idx, col_idx].set_ylim(global_ylim)

                # Set titles and labels
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title(group, fontsize=12)
                if col_idx == 0:
                    axes[row_idx, col_idx].set_ylabel(f'{exp_id}', fontsize=12)
                axes[row_idx, col_idx].set_xlabel(f'{metric}', fontsize=10)

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    plot_single_metric_distribution(aggregated_metrics_dict, 'std_dev')
    return (plot_single_metric_distribution,)


@app.cell
def __(mo):
    mo.md("""## For one experiment, compare the attention score""")
    return


@app.cell
def __(
    Path,
    base_path,
    entropy,
    kurtosis,
    load_attention_scores,
    load_probabilities,
    meta_df,
    np,
    pd,
    plt,
    sns,
):
    import yaml

    # Function to load slides_df based on exp_version
    def load_slides_df(exp_version, config_file=base_path / 'experiment_mapping.yaml'):

        # Load the YAML configuration
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Get the mapping dictionary
        experiment_mapping = config.get('experiment_mapping', {})

        # Check if exp_version exists in the mapping
        if exp_version not in experiment_mapping:
            raise ValueError(f"exp_version '{exp_version}' not found in the configuration file.")

        # Get the data_version for the given exp_version
        data_version = experiment_mapping[exp_version]

        # Construct the path and load the slides DataFrame
        file_path = Path(f'{data_version}/slide_master_skin.csv')
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        print(f"Loading slides_df for {exp_version} from {file_path}")
        slides_df = pd.read_csv(file_path)

        return slides_df[slides_df.split=='val'].copy()

    def calculate_patch_metrics(annotation_df, attention_scores_dict):
        """
        Calculate metrics (entropy, std_dev, kurtosis) for each patch and combine them with annotation data.

        Parameters:
        - annotation_df: DataFrame containing 'is_epithelium' and patch metadata.
        - attention_scores_dict: Dictionary of attention scores keyed by slide ID.

        Returns:
        - metrics_df: DataFrame with calculated metrics and 'is_epithelium'.
        """
        data = []
        for slide_id, scores in attention_scores_dict.items():
            if slide_id in annotation_df['slide'].values:
                is_epithelium_values = annotation_df.loc[annotation_df['slide'] == slide_id, 'is_epithelium'].values

                for i, patch_scores in enumerate(scores):  # Iterate over patches
                    patch_probs = patch_scores / patch_scores.sum() if patch_scores.sum() > 0 else np.ones_like(patch_scores) / len(patch_scores)
                    patch_probs = np.clip(patch_probs, 1e-10, None)  # Avoid log(0)

                    data.append({
                        'entropy': entropy(patch_probs, base=2),
                        'std_dev': np.std(patch_scores),
                        'kurtosis': kurtosis(patch_scores),
                        'is_epithelium': is_epithelium_values[i]
                    })

        # Create DataFrame
        metrics_df = pd.DataFrame(data)
        return metrics_df

    def plot_patch_metrics(metrics_df):
        """
        Plot metrics stratified by 'is_epithelium'.

        Parameters:
        - metrics_df: DataFrame containing metrics ('entropy', 'std_dev', 'kurtosis') and 'is_epithelium'.

        Returns:
        - None
        """
        # Melt the metrics DataFrame for easier plotting
        melted_df = metrics_df.melt(id_vars='is_epithelium', var_name='Metric', value_name='Value')

        # Plot using Seaborn
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=melted_df,
            x='Metric',
            y='Value',
            hue='is_epithelium',
            palette='Set2'
        )
        plt.title('Patch-Level Metrics Stratified by "is_epithelium"', fontsize=16)
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.legend(title='Is Epithelium')
        plt.tight_layout()
        plt.show()

    def predict_is_epithelium_s(meta_df, slides_df, trained_model):
        """Predict 'is_epithelium' for slides."""
        if trained_model is not None:
            slides_of_interest = slides_df['slide'].unique()
            meta_df_subset = meta_df[meta_df['slide'].isin(slides_of_interest)].copy()
            X_all = meta_df_subset[['UMAP_D1', 'UMAP_D2']].values
            meta_df_subset['is_epithelium'] = trained_model.predict(X_all)
            # slides_df = slides_df.merge(meta_df_subset[['slide', 'is_epithelium']], on='slide', how='left')
        else:
            print("Warning: No trained model provided; 'is_epithelium' will not be added.")
            slides_df['is_epithelium'] = None
        return meta_df_subset

    def generate_data(
        exp_version, trained_model
    ):
        aggregated_metrics_list = []
        prob_df = load_probabilities(exp_version)

        # set up attention_scores
        slides_df = load_slides_df(exp_version)    
        attention_scores_dict, global_min, global_max = load_attention_scores(exp_version)
        annotation_df = predict_is_epithelium_s(meta_df, slides_df, trained_model)
        return annotation_df

        # metrics_df = calculate_patch_metrics(annotation_df, attention_scores_dict)
        # plot_patch_metrics(metrics_df)
    return (
        calculate_patch_metrics,
        generate_data,
        load_slides_df,
        plot_patch_metrics,
        predict_is_epithelium_s,
        yaml,
    )


@app.cell
def __(generate_data, joblib):
    model_to_use = joblib.load('model_logistic_2025-01-12.joblib')
    annotation_df_use = generate_data('exp1',model_to_use)
    return annotation_df_use, model_to_use


@app.cell
def __(Path, pickle):
    exp_version = 'exp1'
    """Load attention scores dictionary."""
    base_path_test = Path(f'slide_experiments/skin/{exp_version}')
    attention_dict_file = base_path_test / 'attention_scores_dict.pkl'

    with open(attention_dict_file, 'rb') as f:
        attention_scores_dict = pickle.load(f)
    attention_scores_dict
    return (
        attention_dict_file,
        attention_scores_dict,
        base_path_test,
        exp_version,
        f,
    )


@app.cell
def __(attention_scores_dict, demographic_groups, pd):
    # Expand the attention_scores_dict into a DataFrame
    expanded_data = []
    for slide_name, scores in attention_scores_dict.items():
        num_patches = len(scores[0])  # Number of patches in the slide
        for i in range(num_patches):
            expanded_data.append([slide_name] + [scores[j][i] for j in range(5)])

    # Create a DataFrame from expanded data
    attention_scores_df = pd.DataFrame(expanded_data, columns=["slide"] + demographic_groups)
    attention_scores_df
    return (
        attention_scores_df,
        expanded_data,
        i,
        num_patches,
        scores,
        slide_name,
    )


@app.cell
def __(annotation_df_use, attention_scores_df, demographic_groups, pd):
    concatenated_df = pd.concat([annotation_df_use[['slide','race_curated','kmeans_cluster','is_epithelium']].reset_index(drop=True), attention_scores_df[demographic_groups].reset_index(drop=True)], axis=1)
    concatenated_df
    return (concatenated_df,)


@app.cell
def __(load_attention_scores):
    _, global_min, global_max = load_attention_scores('exp1')
    return global_max, global_min


@app.cell
def __(concatenated_df, demographic_groups, global_max, global_min, plt):
    def plot_heatmaps_epi():
        # Normalize the attention scores
        for col in demographic_groups:
            concatenated_df[col] = (concatenated_df[col] - global_min) / (global_max - global_min)

        # Stratify data by 'is_epithelium'
        stratified_data = concatenated_df.groupby('is_epithelium')

        # Plotting
        fig, axes = plt.subplots(1, len(demographic_groups), figsize=(20, 6), sharey=True)
        fig.suptitle("Normalized Attention Scores Stratified by 'is_epithelium'", fontsize=16)

        for i, demographic in enumerate(demographic_groups):
            ax = axes[i]
            for epithelium_status, group in stratified_data:
                ax.boxplot(
                    group[demographic],
                    positions=[epithelium_status],
                    patch_artist=True,
                    labels=[str(epithelium_status)],
                )
            ax.set_title(demographic)
            ax.set_xlabel("'is_epithelium'")
            ax.set_ylabel("Normalized Attention Score" if i == 0 else "")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
        plt.show()

    plot_heatmaps_epi()
    return (plot_heatmaps_epi,)


@app.cell
def __(concatenated_df, demographic_groups, global_max, global_min, plt):
    def plot_attention_histograms():
        """
        Plot histograms with density for normalized attention scores stratified by 
        'is_epithelium', grouped by each slide.
        """
        # Normalize the attention scores
        for col in demographic_groups:
            concatenated_df[col] = (concatenated_df[col] - global_min) / (global_max - global_min)

        # Get unique slides
        slides = concatenated_df['slide'].unique()

        # Plot histograms for each slide
        for slide in slides:
            slide_data = concatenated_df[concatenated_df['slide'] == slide]

            # Create subplots for demographic groups
            fig, axes = plt.subplots(1, len(demographic_groups), figsize=(20, 6), sharey=True)
            fig.suptitle(f"Normalized Attention Scores for Slide: {slide}", fontsize=16)

            for i, demographic in enumerate(demographic_groups):
                ax = axes[i]

                for epithelium_status, group in slide_data.groupby('is_epithelium'):
                    # Plot histogram with density
                    group[demographic].plot(
                        kind='density', 
                        ax=ax, 
                        label=f"is_epithelium={epithelium_status}", 
                        alpha=0.6
                    )

                ax.set_title(demographic)
                ax.set_xlabel("Normalized Attention Score")
                ax.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the title
            plt.show()
    return (plot_attention_histograms,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
