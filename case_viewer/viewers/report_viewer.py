"""
Report Viewer Module
Displays pathology reports, patient metadata, and clinical information.
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import json


class ReportViewer:
    """Handler for viewing pathology reports and patient metadata."""
    
    def __init__(self):
        self.data_root = Path("data")
        self.metadata_file = self.data_root / "self_reported_race" / "skin" / "master_metadata.csv"
    
    def render(self, case_id=None):
        """Render the report viewer interface."""
        
        if case_id is None:
            # Case selection
            case_id = st.text_input("Enter Case/Slide ID:", "")
            
            if not case_id:
                st.info("Please enter a case ID to view reports.")
                return
        
        # Load case metadata
        case_data = self._load_case_metadata(case_id)
        
        if case_data is None:
            st.warning(f"No metadata found for case {case_id}")
            return
        
        # Display in tabs
        tabs = st.tabs(["Patient Info", "Clinical Data", "Diagnosis", "Experiment Info", "Raw Data"])
        
        with tabs[0]:
            self._render_patient_info(case_data)
        
        with tabs[1]:
            self._render_clinical_data(case_data)
        
        with tabs[2]:
            self._render_diagnosis(case_data)
        
        with tabs[3]:
            self._render_experiment_info(case_data)
        
        with tabs[4]:
            self._render_raw_data(case_data)
    
    def _load_case_metadata(self, case_id):
        """Load metadata for a specific case."""
        
        if not self.metadata_file.exists():
            st.warning(f"Metadata file not found: {self.metadata_file}")
            return None
        
        try:
            # Load metadata
            df = pd.read_csv(self.metadata_file)
            
            # Search for case by slide_id or other identifiers
            # Try multiple columns that might contain the case ID
            search_columns = ['slide_id', 'case_id', 'specimen_id', 'accession_number']
            
            case_data = None
            for col in search_columns:
                if col in df.columns:
                    matches = df[df[col].astype(str) == str(case_id)]
                    if len(matches) > 0:
                        case_data = matches.iloc[0].to_dict()
                        break
            
            return case_data
            
        except Exception as e:
            st.error(f"Error loading metadata: {str(e)}")
            return None
    
    def _render_patient_info(self, case_data):
        """Render patient demographic information."""
        st.markdown("### Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Demographics")
            
            # Self-reported race
            if 'race' in case_data or 'self_reported_race' in case_data:
                race = case_data.get('race', case_data.get('self_reported_race', 'N/A'))
                st.metric("Self-reported Race", race)
            
            # Age
            if 'age' in case_data or 'age_at_diagnosis' in case_data:
                age = case_data.get('age', case_data.get('age_at_diagnosis', 'N/A'))
                st.metric("Age", age)
            
            # Sex/Gender
            if 'sex' in case_data or 'gender' in case_data:
                sex = case_data.get('sex', case_data.get('gender', 'N/A'))
                st.metric("Sex", sex)
        
        with col2:
            st.markdown("#### Identifiers")
            
            # Patient ID
            if 'patient_id' in case_data or 'mrn' in case_data:
                patient_id = case_data.get('patient_id', case_data.get('mrn', 'N/A'))
                st.metric("Patient ID", patient_id)
            
            # Case/Slide ID
            if 'slide_id' in case_data or 'case_id' in case_data:
                slide_id = case_data.get('slide_id', case_data.get('case_id', 'N/A'))
                st.metric("Slide ID", slide_id)
            
            # Accession Number
            if 'accession_number' in case_data:
                st.metric("Accession Number", case_data['accession_number'])
    
    def _render_clinical_data(self, case_data):
        """Render clinical information."""
        st.markdown("### Clinical Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Specimen Information")
            
            # Specimen type
            if 'specimen_type' in case_data or 'tissue_type' in case_data:
                specimen = case_data.get('specimen_type', case_data.get('tissue_type', 'N/A'))
                st.metric("Specimen Type", specimen)
            
            # Body site
            if 'body_site' in case_data or 'anatomic_site' in case_data:
                site = case_data.get('body_site', case_data.get('anatomic_site', 'N/A'))
                st.metric("Body Site", site)
            
            # Collection date
            if 'collection_date' in case_data or 'procedure_date' in case_data:
                date = case_data.get('collection_date', case_data.get('procedure_date', 'N/A'))
                st.metric("Collection Date", date)
        
        with col2:
            st.markdown("#### Medical History")
            
            # Relevant history fields
            history_fields = [
                'medical_history',
                'family_history',
                'previous_diagnosis',
                'comorbidities'
            ]
            
            found_history = False
            for field in history_fields:
                if field in case_data and case_data[field] not in [None, '', 'N/A', 'nan']:
                    st.text_area(field.replace('_', ' ').title(), str(case_data[field]), height=100)
                    found_history = True
            
            if not found_history:
                st.info("No medical history data available")
    
    def _render_diagnosis(self, case_data):
        """Render diagnosis and pathology findings."""
        st.markdown("### Diagnosis & Findings")
        
        # Primary diagnosis
        if 'diagnosis' in case_data or 'primary_diagnosis' in case_data:
            diagnosis = case_data.get('diagnosis', case_data.get('primary_diagnosis', 'N/A'))
            st.markdown("#### Primary Diagnosis")
            st.info(diagnosis)
        
        # ICD codes
        st.markdown("#### Diagnostic Codes")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'icd10_code' in case_data:
                st.metric("ICD-10 Code", case_data['icd10_code'])
            
            if 'icd9_code' in case_data:
                st.metric("ICD-9 Code", case_data['icd9_code'])
        
        with col2:
            if 'snomed_code' in case_data:
                st.metric("SNOMED Code", case_data['snomed_code'])
            
            if 'diagnosis_code' in case_data:
                st.metric("Diagnosis Code", case_data['diagnosis_code'])
        
        # Pathology report text
        report_fields = [
            'pathology_report',
            'microscopic_description',
            'gross_description',
            'impression',
            'comment'
        ]
        
        st.markdown("#### Pathology Report")
        for field in report_fields:
            if field in case_data and case_data[field] not in [None, '', 'N/A', 'nan']:
                with st.expander(field.replace('_', ' ').title()):
                    st.text_area(
                        "Content",
                        str(case_data[field]),
                        height=200,
                        key=f"{field}_text",
                        label_visibility="collapsed"
                    )
    
    def _render_experiment_info(self, case_data):
        """Render experiment-related information."""
        st.markdown("### Experiment Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Dataset")
            
            # Experiment version
            if 'experiment' in case_data or 'exp_version' in case_data:
                exp = case_data.get('experiment', case_data.get('exp_version', 'N/A'))
                st.metric("Experiment", exp)
                
                # Show experiment description
                exp_descriptions = {
                    'Exp1': 'Uncurated - All dermatopathology specimens',
                    'Exp2': 'Balanced Disease - Rebalanced for disease confounding',
                    'Exp3': 'Strict ICD Code - Classical dermatopathology cases only'
                }
                
                if exp in exp_descriptions:
                    st.caption(exp_descriptions[exp])
            
            # Dataset split
            if 'split' in case_data or 'fold' in case_data:
                split = case_data.get('split', case_data.get('fold', 'N/A'))
                st.metric("Dataset Split", split)
        
        with col2:
            st.markdown("#### Model Features")
            
            # Encoder
            if 'encoder' in case_data or 'feature_encoder' in case_data:
                encoder = case_data.get('encoder', case_data.get('feature_encoder', 'N/A'))
                st.metric("Feature Encoder", encoder)
            
            # Number of tiles
            if 'n_tiles' in case_data or 'tile_count' in case_data:
                n_tiles = case_data.get('n_tiles', case_data.get('tile_count', 'N/A'))
                st.metric("Number of Tiles", n_tiles)
        
        with col3:
            st.markdown("#### Predictions")
            
            # Predicted race
            if 'predicted_race' in case_data:
                st.metric("Predicted Race", case_data['predicted_race'])
            
            # Prediction confidence
            if 'prediction_score' in case_data or 'confidence' in case_data:
                score = case_data.get('prediction_score', case_data.get('confidence', 'N/A'))
                if score != 'N/A':
                    st.metric("Confidence", f"{float(score):.3f}")
            
            # Model performance
            if 'auc' in case_data:
                st.metric("AUC", f"{float(case_data['auc']):.3f}")
    
    def _render_raw_data(self, case_data):
        """Render all available metadata as raw data."""
        st.markdown("### Raw Metadata")
        
        st.markdown("All available fields for this case:")
        
        # Convert to DataFrame for better display
        data_df = pd.DataFrame([case_data]).T
        data_df.columns = ['Value']
        data_df.index.name = 'Field'
        
        st.dataframe(data_df, use_container_width=True)
        
        # Export option
        st.download_button(
            label="Download as JSON",
            data=json.dumps(case_data, indent=2),
            file_name=f"case_{case_data.get('slide_id', 'unknown')}_metadata.json",
            mime="application/json"
        )
