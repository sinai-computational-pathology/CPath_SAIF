"""
Data visualization dashboard for hierarchical pathology data
Displays patient demographics, case distributions, specimen/slide statistics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import SAIFDatabase

st.set_page_config(page_title="Data Dashboard", page_icon="📊", layout="wide")

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "saif.duckdb"


@st.cache_resource
def get_database():
    """Get database connection."""
    return SAIFDatabase(str(DB_PATH))


def load_patient_demographics(db: SAIFDatabase) -> pd.DataFrame:
    """Load patient demographics with parsed metadata."""
    query = """
    SELECT 
        mrn,
        age,
        gender,
        race,
        metadata,
        created_at
    FROM patients
    ORDER BY mrn
    """
    df = db.conn.execute(query).fetchdf()
    
    # Parse metadata JSON
    if 'metadata' in df.columns and not df.empty:
        import ast
        for idx, row in df.iterrows():
            if pd.notna(row['metadata']):
                try:
                    meta = ast.literal_eval(row['metadata'])
                    for key, val in meta.items():
                        df.at[idx, key] = val
                except:
                    pass
    
    return df


def load_case_summary(db: SAIFDatabase) -> pd.DataFrame:
    """Load case-level summary with counts."""
    query = """
    SELECT 
        c.accession_no,
        c.mrn,
        c.recv_date,
        c.metadata,
        COUNT(DISTINCT s.specimen_id) as specimen_count,
        COUNT(DISTINCT w.wsi_id) as slide_count
    FROM cases c
    LEFT JOIN specimens s ON c.accession_no = s.accession_no
    LEFT JOIN wsi_images w ON s.specimen_id = w.specimen_id
    GROUP BY c.accession_no, c.mrn, c.recv_date, c.metadata
    ORDER BY c.recv_date DESC
    """
    return db.conn.execute(query).fetchdf()


def load_specimen_summary(db: SAIFDatabase) -> pd.DataFrame:
    """Load specimen-level summary."""
    query = """
    SELECT 
        s.specimen_id,
        s.accession_no,
        s.specimen_number,
        s.specimen_organ,
        COUNT(w.wsi_id) as slide_count
    FROM specimens s
    LEFT JOIN wsi_images w ON s.specimen_id = w.specimen_id
    GROUP BY s.specimen_id, s.accession_no, s.specimen_number, s.specimen_organ
    ORDER BY s.accession_no, s.specimen_number
    """
    return db.conn.execute(query).fetchdf()


def load_slide_summary(db: SAIFDatabase) -> pd.DataFrame:
    """Load slide-level data."""
    query = """
    SELECT 
        w.wsi_id,
        w.specimen_id,
        w.file_name,
        w.file_path,
        w.stain_type,
        w.metadata,
        s.accession_no,
        s.specimen_number,
        c.mrn
    FROM wsi_images w
    JOIN specimens s ON w.specimen_id = s.specimen_id
    JOIN cases c ON s.accession_no = c.accession_no
    ORDER BY c.mrn, s.accession_no, s.specimen_number
    """
    return db.conn.execute(query).fetchdf()


def main():
    st.title("📊 SAIF Data Dashboard")
    st.markdown("Hierarchical visualization of pathology data: Patients → Cases → Specimens → Slides")
    
    # Check database exists
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}")
        st.info("Run `python init_db.py` to initialize the database")
        return
    
    try:
        db = get_database()
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return
    
    # Overall statistics
    st.header("📈 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        patient_count = db.conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        st.metric("Total Patients", f"{patient_count:,}")
    
    with col2:
        case_count = db.conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        st.metric("Total Cases", f"{case_count:,}")
    
    with col3:
        specimen_count = db.conn.execute("SELECT COUNT(*) FROM specimens").fetchone()[0]
        st.metric("Total Specimens", f"{specimen_count:,}")
    
    with col4:
        slide_count = db.conn.execute("SELECT COUNT(*) FROM wsi_images").fetchone()[0]
        st.metric("Total Slides", f"{slide_count:,}")
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Patient Demographics", 
        "📋 Case Distribution", 
        "🔬 Specimen Analysis",
        "🖼️ Slide Overview"
    ])
    
    # Tab 1: Patient Demographics
    with tab1:
        st.subheader("Patient Demographics")
        
        patients_df = load_patient_demographics(db)
        
        if patients_df.empty:
            st.info("No patient data available")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                if 'age' in patients_df.columns and patients_df['age'].notna().any():
                    fig_age = px.histogram(
                        patients_df, 
                        x='age',
                        title='Age Distribution',
                        labels={'age': 'Age', 'count': 'Number of Patients'},
                        nbins=20
                    )
                    fig_age.update_layout(showlegend=False)
                    st.plotly_chart(fig_age, use_container_width=True)
                
                # Gender distribution
                if 'gender' in patients_df.columns and patients_df['gender'].notna().any():
                    gender_counts = patients_df['gender'].value_counts()
                    fig_gender = px.pie(
                        values=gender_counts.values,
                        names=gender_counts.index,
                        title='Gender Distribution'
                    )
                    st.plotly_chart(fig_gender, use_container_width=True)
            
            with col2:
                # Race distribution
                if 'curated_race' in patients_df.columns and patients_df['curated_race'].notna().any():
                    race_counts = patients_df['curated_race'].value_counts()
                    fig_race = px.bar(
                        x=race_counts.index,
                        y=race_counts.values,
                        title='Race/Ethnicity Distribution',
                        labels={'x': 'Race/Ethnicity', 'y': 'Count'}
                    )
                    fig_race.update_layout(showlegend=False)
                    st.plotly_chart(fig_race, use_container_width=True)
                
                # Age group distribution
                if 'age_group' in patients_df.columns and patients_df['age_group'].notna().any():
                    age_group_counts = patients_df['age_group'].value_counts()
                    fig_age_group = px.bar(
                        x=age_group_counts.index,
                        y=age_group_counts.values,
                        title='Age Group Distribution',
                        labels={'x': 'Age Group', 'y': 'Count'}
                    )
                    fig_age_group.update_layout(showlegend=False)
                    st.plotly_chart(fig_age_group, use_container_width=True)
            
            # Patient table
            st.subheader("Patient Details")
            display_cols = ['mrn', 'age', 'gender', 'curated_race', 'age_group']
            display_cols = [c for c in display_cols if c in patients_df.columns]
            st.dataframe(patients_df[display_cols], use_container_width=True, hide_index=True)
    
    # Tab 2: Case Distribution
    with tab2:
        st.subheader("Case Distribution")
        
        cases_df = load_case_summary(db)
        
        if cases_df.empty:
            st.info("No case data available")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Cases over time
                if 'recv_date' in cases_df.columns and cases_df['recv_date'].notna().any():
                    cases_df['recv_date'] = pd.to_datetime(cases_df['recv_date'])
                    cases_df['year_month'] = cases_df['recv_date'].dt.to_period('M').astype(str)
                    
                    time_counts = cases_df.groupby('year_month').size().reset_index(name='count')
                    
                    fig_time = px.line(
                        time_counts,
                        x='year_month',
                        y='count',
                        title='Cases Over Time',
                        labels={'year_month': 'Month', 'count': 'Number of Cases'}
                    )
                    fig_time.update_layout(showlegend=False)
                    st.plotly_chart(fig_time, use_container_width=True)
                
                # Specimens per case distribution
                if 'specimen_count' in cases_df.columns:
                    fig_spec = px.histogram(
                        cases_df,
                        x='specimen_count',
                        title='Specimens per Case Distribution',
                        labels={'specimen_count': 'Number of Specimens', 'count': 'Cases'},
                        nbins=10
                    )
                    fig_spec.update_layout(showlegend=False)
                    st.plotly_chart(fig_spec, use_container_width=True)
            
            with col2:
                # Cases by year
                if 'recv_date' in cases_df.columns and cases_df['recv_date'].notna().any():
                    cases_df['year'] = cases_df['recv_date'].dt.year
                    year_counts = cases_df['year'].value_counts().sort_index()
                    
                    fig_year = px.bar(
                        x=year_counts.index,
                        y=year_counts.values,
                        title='Cases by Year',
                        labels={'x': 'Year', 'y': 'Number of Cases'}
                    )
                    fig_year.update_layout(showlegend=False)
                    st.plotly_chart(fig_year, use_container_width=True)
                
                # Slides per case distribution
                if 'slide_count' in cases_df.columns:
                    fig_slides = px.histogram(
                        cases_df,
                        x='slide_count',
                        title='Slides per Case Distribution',
                        labels={'slide_count': 'Number of Slides', 'count': 'Cases'},
                        nbins=15
                    )
                    fig_slides.update_layout(showlegend=False)
                    st.plotly_chart(fig_slides, use_container_width=True)
            
            # Case summary table
            st.subheader("Case Summary")
            display_cols = ['accession_no', 'mrn', 'recv_date', 'specimen_count', 'slide_count']
            display_cols = [c for c in display_cols if c in cases_df.columns]
            st.dataframe(cases_df[display_cols], use_container_width=True, hide_index=True)
    
    # Tab 3: Specimen Analysis
    with tab3:
        st.subheader("Specimen Analysis")
        
        specimens_df = load_specimen_summary(db)
        
        if specimens_df.empty:
            st.info("No specimen data available")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Organ distribution
                if 'specimen_organ' in specimens_df.columns and specimens_df['specimen_organ'].notna().any():
                    organ_counts = specimens_df['specimen_organ'].value_counts()
                    fig_organ = px.bar(
                        x=organ_counts.values,
                        y=organ_counts.index,
                        orientation='h',
                        title='Specimen by Organ',
                        labels={'x': 'Count', 'y': 'Organ'}
                    )
                    fig_organ.update_layout(showlegend=False)
                    st.plotly_chart(fig_organ, use_container_width=True)
            
            with col2:
                # Slides per specimen
                if 'slide_count' in specimens_df.columns:
                    fig_slides_spec = px.histogram(
                        specimens_df,
                        x='slide_count',
                        title='Slides per Specimen Distribution',
                        labels={'slide_count': 'Number of Slides', 'count': 'Specimens'},
                        nbins=10
                    )
                    fig_slides_spec.update_layout(showlegend=False)
                    st.plotly_chart(fig_slides_spec, use_container_width=True)
            
            # Specimen table
            st.subheader("Specimen Details")
            st.dataframe(specimens_df, use_container_width=True, hide_index=True)
    
    # Tab 4: Slide Overview
    with tab4:
        st.subheader("Slide Overview")
        
        slides_df = load_slide_summary(db)
        
        if slides_df.empty:
            st.info("No slide data available")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Stain type distribution
                if 'stain_type' in slides_df.columns and slides_df['stain_type'].notna().any():
                    stain_counts = slides_df['stain_type'].value_counts()
                    fig_stain = px.pie(
                        values=stain_counts.values,
                        names=stain_counts.index,
                        title='Stain Type Distribution'
                    )
                    st.plotly_chart(fig_stain, use_container_width=True)
            
            with col2:
                # Slides per patient
                if 'mrn' in slides_df.columns:
                    slides_per_patient = slides_df.groupby('mrn').size().reset_index(name='count')
                    fig_patient_slides = px.histogram(
                        slides_per_patient,
                        x='count',
                        title='Slides per Patient Distribution',
                        labels={'count': 'Number of Slides', 'y': 'Patients'},
                        nbins=20
                    )
                    fig_patient_slides.update_layout(showlegend=False)
                    st.plotly_chart(fig_patient_slides, use_container_width=True)
            
            # Slide table with search
            st.subheader("Slide Details")
            
            # Search filter
            search_term = st.text_input("🔍 Search by Barcode, MRN, or Accession No", "")
            
            if search_term:
                mask = (
                    slides_df['wsi_id'].astype(str).str.contains(search_term, case=False, na=False) |
                    slides_df['mrn'].astype(str).str.contains(search_term, case=False, na=False) |
                    slides_df['accession_no'].astype(str).str.contains(search_term, case=False, na=False)
                )
                filtered_df = slides_df[mask]
                st.info(f"Found {len(filtered_df)} slides matching '{search_term}'")
            else:
                filtered_df = slides_df
            
            display_cols = ['wsi_id', 'mrn', 'accession_no', 'specimen_number', 
                          'stain_type', 'file_name']
            display_cols = [c for c in display_cols if c in filtered_df.columns]
            
            st.dataframe(
                filtered_df[display_cols].head(100), 
                use_container_width=True, 
                hide_index=True
            )
            
            if len(filtered_df) > 100:
                st.info(f"Showing first 100 of {len(filtered_df)} slides. Use search to narrow results.")


if __name__ == "__main__":
    main()
