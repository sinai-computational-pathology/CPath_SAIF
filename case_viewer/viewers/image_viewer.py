"""
Image Viewer Module
Displays whole-slide images, attention maps, and tile-level visualizations.
"""

import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ImageViewer:
    """Handler for viewing histological images and attention maps."""
    
    def __init__(self):
        self.data_root = Path("data")
        
    def render(self, case_id=None):
        """Render the image viewer interface."""
        
        if case_id is None:
            # Case selection
            case_id = st.text_input("Enter Case/Slide ID:", "")
            
            if not case_id:
                st.info("Please enter a case ID to view images.")
                return
        
        # Display options
        st.subheader("Display Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_thumbnail = st.checkbox("Show Thumbnail", value=True)
        with col2:
            show_attention = st.checkbox("Show Attention Map", value=True)
        with col3:
            show_tiles = st.checkbox("Show High-Attention Tiles", value=False)
        
        # Encoder selection for attention maps
        if show_attention:
            encoder = st.selectbox(
                "Select Encoder:",
                ["SP22M", "UNI", "GigaPath", "Virchow"]
            )
        else:
            encoder = "SP22M"
        
        # Main image display
        st.divider()
        
        tabs = st.tabs(["Slide Overview", "Attention Maps", "Tile Analysis", "UMAP Projection"])
        
        with tabs[0]:
            self._render_slide_overview(case_id, show_thumbnail)
        
        with tabs[1]:
            self._render_attention_maps(case_id, encoder)
        
        with tabs[2]:
            self._render_tile_analysis(case_id, encoder)
        
        with tabs[3]:
            self._render_umap_projection(case_id, encoder)
    
    def _render_slide_overview(self, case_id, show_thumbnail):
        """Render slide overview with thumbnail."""
        st.markdown("### Slide Overview")
        
        # Try to load slide thumbnail
        thumbnail_path = self._find_slide_thumbnail(case_id)
        
        if thumbnail_path and thumbnail_path.exists():
            try:
                img = Image.open(thumbnail_path)
                st.image(img, caption=f"Slide {case_id}", use_container_width=True)
                
                # Display image info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Width", f"{img.width} px")
                with col2:
                    st.metric("Height", f"{img.height} px")
                with col3:
                    st.metric("Aspect Ratio", f"{img.width/img.height:.2f}")
            except Exception as e:
                st.error(f"Error loading thumbnail: {str(e)}")
        else:
            st.info(f"No thumbnail found for case {case_id}")
            st.markdown("""
            **Expected location:**
            - `data/slides/{case_id}.png` or
            - `data/thumbnails/{case_id}.jpg`
            """)
    
    def _render_attention_maps(self, case_id, encoder):
        """Render attention maps for different race groups."""
        st.markdown("### Attention Maps")
        
        st.markdown("""
        Attention maps highlight regions of the slide that the model focuses on 
        when making predictions for each demographic group.
        """)
        
        # Race group selection
        race_groups = ["White", "Black", "Hispanic/Latino", "Asian", "Other"]
        selected_groups = st.multiselect(
            "Select race groups to display:",
            race_groups,
            default=["White", "Black"]
        )
        
        if not selected_groups:
            st.warning("Please select at least one race group.")
            return
        
        # Try to load attention maps
        attention_dir = Path(f"slide_experiments/skin/{encoder}/attention_maps")
        
        if not attention_dir.exists():
            st.warning(f"Attention maps directory not found: {attention_dir}")
            st.info("Run `get_attention.py` to generate attention maps first.")
            return
        
        # Display attention maps in grid
        cols = st.columns(len(selected_groups))
        
        for idx, (col, race) in enumerate(zip(cols, selected_groups)):
            with col:
                st.markdown(f"**{race}**")
                
                # Try to find attention map
                attn_file = attention_dir / f"{case_id}_{race.replace('/', '_')}_attention.png"
                
                if attn_file.exists():
                    try:
                        img = Image.open(attn_file)
                        st.image(img, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.info("No attention map found")
    
    def _render_tile_analysis(self, case_id, encoder):
        """Render tile-level analysis with high-attention regions."""
        st.markdown("### Tile-Level Analysis")
        
        st.markdown("""
        View individual tiles from the slide, sorted by attention score.
        This helps identify specific tissue structures the model focuses on.
        """)
        
        # Load tile coordinates and attention scores
        coord_path = Path(f"data/{encoder}/coordinates/{case_id}.csv")
        
        if not coord_path.exists():
            st.warning(f"Coordinate file not found: {coord_path}")
            st.info("Tile coordinates should be generated during feature extraction.")
            return
        
        try:
            # Load coordinates
            coords_df = pd.read_csv(coord_path)
            
            st.success(f"Loaded {len(coords_df)} tiles")
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tiles", len(coords_df))
            with col2:
                if 'attention' in coords_df.columns:
                    st.metric("Mean Attention", f"{coords_df['attention'].mean():.4f}")
            with col3:
                if 'attention' in coords_df.columns:
                    st.metric("Max Attention", f"{coords_df['attention'].max():.4f}")
            
            # Scatter plot of tile locations colored by attention
            if 'x' in coords_df.columns and 'y' in coords_df.columns:
                fig = px.scatter(
                    coords_df,
                    x='x',
                    y='y',
                    color='attention' if 'attention' in coords_df.columns else None,
                    title="Tile Locations on Slide",
                    labels={'x': 'X Coordinate', 'y': 'Y Coordinate', 'attention': 'Attention Score'},
                    hover_data=['x', 'y']
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
                
            # Show top tiles by attention
            if 'attention' in coords_df.columns:
                st.markdown("#### Top Tiles by Attention Score")
                
                n_tiles = st.slider("Number of top tiles to display:", 5, 50, 10)
                top_tiles = coords_df.nlargest(n_tiles, 'attention')
                
                st.dataframe(top_tiles, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading tile data: {str(e)}")
    
    def _render_umap_projection(self, case_id, encoder):
        """Render UMAP projection for tiles from this slide."""
        st.markdown("### UMAP Projection")
        
        st.markdown("""
        Visualize the distribution of tiles from this slide in the learned 
        feature space using UMAP dimensionality reduction.
        """)
        
        # Check for UMAP data
        umap_file = Path("Emb_Viz/marimo_metadata.csv")
        
        if not umap_file.exists():
            st.warning("UMAP data not found. Run the UMAP visualization pipeline first.")
            st.info("""
            To generate UMAP projections:
            1. Run `python Emb_Viz/prepare_meta_source.py`
            2. Run `marimo run Emb_Viz/tile_representation_umap.py`
            """)
            return
        
        try:
            # Load UMAP data
            umap_df = pd.read_csv(umap_file)
            
            # Filter for current slide
            if 'slide_id' in umap_df.columns:
                slide_data = umap_df[umap_df['slide_id'] == case_id]
                
                if len(slide_data) > 0:
                    st.success(f"Found {len(slide_data)} tiles in UMAP space")
                    
                    # Create UMAP scatter plot
                    if 'UMAP_D1' in slide_data.columns and 'UMAP_D2' in slide_data.columns:
                        fig = px.scatter(
                            slide_data,
                            x='UMAP_D1',
                            y='UMAP_D2',
                            color='tissue_type' if 'tissue_type' in slide_data.columns else None,
                            title=f"UMAP Projection for Slide {case_id}",
                            hover_data=slide_data.columns
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No tiles found for slide {case_id} in UMAP data")
            else:
                st.warning("Slide ID column not found in UMAP data")
                
        except Exception as e:
            st.error(f"Error loading UMAP data: {str(e)}")
    
    def _find_slide_thumbnail(self, case_id):
        """Find thumbnail for a given case ID."""
        # Search in multiple possible locations
        possible_paths = [
            Path(f"data/slides/{case_id}.png"),
            Path(f"data/slides/{case_id}.jpg"),
            Path(f"data/thumbnails/{case_id}.png"),
            Path(f"data/thumbnails/{case_id}.jpg"),
            Path(f"figures/slides/{case_id}.png"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
