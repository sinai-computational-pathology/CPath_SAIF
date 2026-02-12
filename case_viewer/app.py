"""
SAIF Case Viewer - Main Application
A lightweight data viewer for pathology reports and histological images.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import DataLoader
from viewers.image_viewer import ImageViewer
from viewers.report_viewer import ReportViewer


def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="SAIF Case Viewer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🔬 SAIF Case Viewer")
    st.markdown("""
    **Self-reported race prediction in dermatopathology using AI - Data Viewer**
    
    This viewer allows you to explore:
    - **Pathology Reports**: View patient reports and metadata
    - **Histological Images**: Browse whole-slide images and attention maps
    """)
    
    # Initialize data loader
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    view_mode = st.sidebar.radio(
        "Select View Mode:",
        ["Home", "Case Browser", "Image Viewer", "Report Viewer", "Combined View"]
    )
    
    # Main content area
    if view_mode == "Home":
        show_home()
    elif view_mode == "Case Browser":
        show_case_browser()
    elif view_mode == "Image Viewer":
        ImageViewer().render()
    elif view_mode == "Report Viewer":
        ReportViewer().render()
    elif view_mode == "Combined View":
        show_combined_view()


def show_home():
    """Display home page with project information."""
    st.header("Welcome to SAIF Case Viewer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Project Overview")
        st.markdown("""
        This project investigates fairness and bias in computational pathology using AI.
        We explore whether deep learning models can predict self-reported race from 
        digitized dermatopathology slides.
        
        **Key Features:**
        - Multi-site dataset with racially diverse population
        - Attention-based mechanism for morphological feature discovery
        - Three dataset curation strategies to control confounding factors
        """)
        
    with col2:
        st.subheader("📁 Dataset Summary")
        st.markdown("""
        **Total Slides:** 5,266  
        **Total Patients:** 2,471
        
        **Self-reported Race Distribution:**
        - White: 40.8%
        - Black: 19.3%
        - Hispanic/Latino: 16.5%
        - Asian: 13.1%
        - Other: 10.3%
        """)
    
    st.divider()
    
    # Quick start guide
    st.subheader("🚀 Quick Start")
    st.markdown("""
    1. **Case Browser**: Browse all cases with filters
    2. **Image Viewer**: View whole-slide images and attention maps
    3. **Report Viewer**: Read pathology reports and metadata
    4. **Combined View**: View images and reports side-by-side
    """)


def show_case_browser():
    """Display case browser with filtering options."""
    st.header("Case Browser")
    
    # Filters in sidebar
    st.sidebar.subheader("Filters")
    
    # Race filter
    races = ["All", "White", "Black", "Hispanic/Latino", "Asian", "Other"]
    selected_race = st.sidebar.multiselect(
        "Self-reported Race:",
        races[1:],
        default=[]
    )
    
    # Experiment filter
    experiments = ["All", "Exp1", "Exp2", "Exp3"]
    selected_exp = st.sidebar.selectbox("Experiment:", experiments)
    
    # Encoder filter
    encoders = ["All", "SP22M", "UNI", "GigaPath", "Virchow"]
    selected_encoder = st.sidebar.selectbox("Encoder:", encoders)
    
    # Main content
    st.markdown("### Available Cases")
    
    # Search box
    search_query = st.text_input("🔍 Search by Case ID or Patient ID:", "")
    
    # Load and filter data
    data_loader = st.session_state.data_loader
    cases = data_loader.get_cases(
        race=selected_race if selected_race else None,
        experiment=selected_exp if selected_exp != "All" else None,
        encoder=selected_encoder if selected_encoder != "All" else None,
        search_query=search_query
    )
    
    # Display results
    st.info(f"Found {len(cases)} cases matching criteria")
    
    if len(cases) > 0:
        # Display as table with selection
        for idx, case in enumerate(cases[:50]):  # Limit to first 50
            with st.expander(f"Case {case['case_id']} - {case['race']} - {case.get('diagnosis', 'N/A')}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Patient ID:** {case['patient_id']}")
                    st.write(f"**Self-reported Race:** {case['race']}")
                
                with col2:
                    st.write(f"**Experiment:** {case.get('experiment', 'N/A')}")
                    st.write(f"**Encoder:** {case.get('encoder', 'N/A')}")
                
                with col3:
                    if st.button("View Details", key=f"view_{idx}"):
                        st.session_state.selected_case = case
                        st.rerun()
    else:
        st.warning("No cases found matching the selected criteria.")


def show_combined_view():
    """Display combined view with images and reports side-by-side."""
    st.header("Combined View")
    
    if 'selected_case' not in st.session_state:
        st.info("Please select a case from the Case Browser first.")
        return
    
    case = st.session_state.selected_case
    
    # Case information header
    st.subheader(f"Case {case['case_id']}")
    st.markdown(f"**Patient ID:** {case['patient_id']} | **Race:** {case['race']}")
    
    st.divider()
    
    # Two columns for image and report
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### Histological Image")
        ImageViewer().render(case_id=case['case_id'])
    
    with col2:
        st.markdown("### Pathology Report")
        ReportViewer().render(case_id=case['case_id'])


if __name__ == "__main__":
    main()
