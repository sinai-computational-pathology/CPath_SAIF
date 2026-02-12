"""
Data Loader Utility
Handles loading and filtering case data from various sources.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import streamlit as st


class DataLoader:
    """Centralized data loading and management for the case viewer."""
    
    def __init__(self):
        """Initialize data loader with default paths."""
        self.data_root = Path("data")
        self.metadata_file = self.data_root / "self_reported_race" / "skin" / "master_metadata.csv"
        self._metadata_cache = None
        
    @st.cache_data(ttl=3600)
    def load_metadata(_self) -> pd.DataFrame:
        """Load and cache metadata file.
        
        Returns:
            DataFrame with case metadata
        """
        if _self._metadata_cache is not None:
            return _self._metadata_cache
            
        if not _self.metadata_file.exists():
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'case_id', 'patient_id', 'slide_id', 'race', 
                'diagnosis', 'experiment', 'encoder'
            ])
        
        try:
            df = pd.read_csv(_self.metadata_file)
            _self._metadata_cache = df
            return df
        except Exception as e:
            st.error(f"Error loading metadata: {str(e)}")
            return pd.DataFrame()
    
    def get_cases(
        self,
        race: Optional[List[str]] = None,
        experiment: Optional[str] = None,
        encoder: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get filtered list of cases.
        
        Args:
            race: Filter by self-reported race (list of races)
            experiment: Filter by experiment version (Exp1, Exp2, Exp3)
            encoder: Filter by encoder (SP22M, UNI, GigaPath, Virchow)
            search_query: Search string for case/patient ID
            limit: Maximum number of cases to return
            
        Returns:
            List of case dictionaries
        """
        df = self.load_metadata()
        
        if df.empty:
            return []
        
        # Apply filters
        if race:
            # Try different column names for race
            race_col = self._find_column(df, ['race', 'self_reported_race', 'demographic_race'])
            if race_col:
                df = df[df[race_col].isin(race)]
        
        if experiment:
            exp_col = self._find_column(df, ['experiment', 'exp_version', 'dataset'])
            if exp_col:
                df = df[df[exp_col] == experiment]
        
        if encoder:
            enc_col = self._find_column(df, ['encoder', 'feature_encoder', 'model'])
            if enc_col:
                df = df[df[enc_col] == encoder]
        
        if search_query:
            # Search across multiple ID columns
            id_cols = ['case_id', 'patient_id', 'slide_id', 'specimen_id', 'accession_number']
            mask = pd.Series([False] * len(df))
            
            for col in id_cols:
                if col in df.columns:
                    mask |= df[col].astype(str).str.contains(search_query, case=False, na=False)
            
            df = df[mask]
        
        # Limit results
        df = df.head(limit)
        
        # Convert to list of dictionaries
        cases = df.to_dict('records')
        
        # Standardize field names
        standardized_cases = []
        for case in cases:
            std_case = self._standardize_case(case)
            standardized_cases.append(std_case)
        
        return standardized_cases
    
    def get_case_by_id(self, case_id: str) -> Optional[Dict]:
        """Get a specific case by ID.
        
        Args:
            case_id: Case/slide identifier
            
        Returns:
            Case dictionary or None if not found
        """
        df = self.load_metadata()
        
        if df.empty:
            return None
        
        # Search in multiple ID columns
        id_cols = ['case_id', 'slide_id', 'specimen_id', 'accession_number']
        
        for col in id_cols:
            if col in df.columns:
                matches = df[df[col].astype(str) == str(case_id)]
                if len(matches) > 0:
                    case = matches.iloc[0].to_dict()
                    return self._standardize_case(case)
        
        return None
    
    def get_race_distribution(self, experiment: Optional[str] = None) -> Dict[str, int]:
        """Get distribution of cases by race.
        
        Args:
            experiment: Filter by experiment (optional)
            
        Returns:
            Dictionary mapping race to count
        """
        df = self.load_metadata()
        
        if df.empty:
            return {}
        
        if experiment:
            exp_col = self._find_column(df, ['experiment', 'exp_version'])
            if exp_col:
                df = df[df[exp_col] == experiment]
        
        race_col = self._find_column(df, ['race', 'self_reported_race'])
        if race_col:
            return df[race_col].value_counts().to_dict()
        
        return {}
    
    def get_available_encoders(self) -> List[str]:
        """Get list of available encoders.
        
        Returns:
            List of encoder names
        """
        df = self.load_metadata()
        
        if df.empty:
            return ["SP22M", "UNI", "GigaPath", "Virchow"]  # Default list
        
        enc_col = self._find_column(df, ['encoder', 'feature_encoder'])
        if enc_col:
            return sorted(df[enc_col].dropna().unique().tolist())
        
        return ["SP22M", "UNI", "GigaPath", "Virchow"]
    
    def get_available_experiments(self) -> List[str]:
        """Get list of available experiments.
        
        Returns:
            List of experiment names
        """
        df = self.load_metadata()
        
        if df.empty:
            return ["Exp1", "Exp2", "Exp3"]  # Default list
        
        exp_col = self._find_column(df, ['experiment', 'exp_version'])
        if exp_col:
            return sorted(df[exp_col].dropna().unique().tolist())
        
        return ["Exp1", "Exp2", "Exp3"]
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column name from candidates.
        
        Args:
            df: DataFrame to search
            candidates: List of candidate column names
            
        Returns:
            First matching column name or None
        """
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _standardize_case(self, case: Dict) -> Dict:
        """Standardize case field names.
        
        Args:
            case: Original case dictionary
            
        Returns:
            Standardized case dictionary
        """
        # Map alternative field names to standard names
        field_mappings = {
            'case_id': ['case_id', 'slide_id', 'specimen_id'],
            'patient_id': ['patient_id', 'mrn', 'patient_number'],
            'race': ['race', 'self_reported_race', 'demographic_race'],
            'experiment': ['experiment', 'exp_version', 'dataset'],
            'encoder': ['encoder', 'feature_encoder', 'model'],
            'diagnosis': ['diagnosis', 'primary_diagnosis', 'final_diagnosis']
        }
        
        std_case = case.copy()
        
        for std_name, alternatives in field_mappings.items():
            if std_name not in std_case or pd.isna(std_case[std_name]):
                for alt in alternatives:
                    if alt in case and not pd.isna(case[alt]):
                        std_case[std_name] = case[alt]
                        break
        
        return std_case
    
    @staticmethod
    def load_attention_scores(slide_id: str, encoder: str = "SP22M") -> Optional[pd.DataFrame]:
        """Load attention scores for a slide.
        
        Args:
            slide_id: Slide identifier
            encoder: Encoder name
            
        Returns:
            DataFrame with attention scores or None
        """
        attention_file = Path(f"slide_experiments/skin/{encoder}/attention/{slide_id}_attention.csv")
        
        if not attention_file.exists():
            return None
        
        try:
            return pd.read_csv(attention_file)
        except Exception:
            return None
    
    @staticmethod
    def load_tile_coordinates(slide_id: str, encoder: str = "SP22M") -> Optional[pd.DataFrame]:
        """Load tile coordinates for a slide.
        
        Args:
            slide_id: Slide identifier
            encoder: Encoder name
            
        Returns:
            DataFrame with tile coordinates or None
        """
        coord_file = Path(f"data/{encoder}/coordinates/{slide_id}.csv")
        
        if not coord_file.exists():
            return None
        
        try:
            return pd.read_csv(coord_file)
        except Exception:
            return None
    
    @staticmethod
    def load_features(slide_id: str, encoder: str = "SP22M") -> Optional[np.ndarray]:
        """Load feature embeddings for a slide.
        
        Args:
            slide_id: Slide identifier
            encoder: Encoder name
            
        Returns:
            Numpy array with features or None
        """
        import torch
        
        feat_file = Path(f"data/{encoder}/features/{slide_id}.pth")
        
        if not feat_file.exists():
            return None
        
        try:
            features = torch.load(feat_file)
            if isinstance(features, torch.Tensor):
                return features.numpy()
            return features
        except Exception:
            return None
