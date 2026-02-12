"""
DuckDB Database Manager for SAIF Pathology Data
Handles patient, case, specimen, WSI, and experiment data
"""

import duckdb
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Optional, Union


class SAIFDatabase:
    """Main database manager for SAIF project data."""
    
    def __init__(self, db_path: str = "data/saif.duckdb"):
        """Initialize database connection.
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Create database schema if it doesn't exist."""
        
        # Patients table - core demographic and clinical data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                mrn TEXT PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                race TEXT,
                ethnicity TEXT,
                diagnosis_date DATE,
                diagnosis_description TEXT,
                metadata JSON,  -- Flexible JSON for additional fields
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Cases table - pathology reports and case-level data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                accession_no TEXT PRIMARY KEY,
                mrn TEXT,
                recv_date TIMESTAMP,
                report_text TEXT,
                case_type TEXT,
                organ_system TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (mrn) REFERENCES patients(mrn)
            )
        """)
        
        # Specimens table - specimen-level data extracted from reports
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS specimens (
                specimen_id TEXT PRIMARY KEY,
                accession_no TEXT,
                specimen_number TEXT,  -- A, B, C, etc.
                specimen_source TEXT,
                specimen_organ TEXT,
                specimen_type TEXT,
                diagnosis TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (accession_no) REFERENCES cases(accession_no)
            )
        """)
        
        # WSI images table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS wsi_images (
                wsi_id TEXT PRIMARY KEY,
                specimen_id TEXT,
                file_path TEXT,
                file_name TEXT,
                magnification TEXT,
                stain_type TEXT,
                width INTEGER,
                height INTEGER,
                file_size_mb FLOAT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (specimen_id) REFERENCES specimens(specimen_id)
            )
        """)
        
        # Text extraction results (LLM outputs)
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_extraction_id START 1;
            CREATE TABLE IF NOT EXISTS text_extractions (
                extraction_id INTEGER PRIMARY KEY DEFAULT nextval('seq_extraction_id'),
                accession_no TEXT,
                model_name TEXT,
                extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                results JSON,  -- Full extraction results
                status TEXT,  -- 'success', 'failed', 'partial'
                metadata JSON,
                FOREIGN KEY (accession_no) REFERENCES cases(accession_no)
            )
        """)
        
        # Data splits for ML experiments
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_split_id START 1;
            CREATE TABLE IF NOT EXISTS data_splits (
                split_id INTEGER PRIMARY KEY DEFAULT nextval('seq_split_id'),
                split_name TEXT UNIQUE,
                description TEXT,
                criteria JSON,  -- Curation conditions
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                total_cases INTEGER,
                train_count INTEGER,
                val_count INTEGER,
                test_count INTEGER
            )
        """)
        
        # Data split members - which cases belong to which split
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_split_members (
                split_id INTEGER,
                accession_no TEXT,
                split_type TEXT,  -- 'train', 'val', 'test'
                PRIMARY KEY (split_id, accession_no),
                FOREIGN KEY (split_id) REFERENCES data_splits(split_id),
                FOREIGN KEY (accession_no) REFERENCES cases(accession_no)
            )
        """)
        
        # Experiments tracking
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_experiment_id START 1;
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY DEFAULT nextval('seq_experiment_id'),
                experiment_name TEXT UNIQUE,
                split_id INTEGER,
                model_type TEXT,
                model_config JSON,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,  -- 'running', 'completed', 'failed'
                description TEXT,
                metadata JSON,
                FOREIGN KEY (split_id) REFERENCES data_splits(split_id)
            )
        """)
        
        # Model results - overall and subgroup performance
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_result_id START 1;
            CREATE TABLE IF NOT EXISTS model_results (
                result_id INTEGER PRIMARY KEY DEFAULT nextval('seq_result_id'),
                experiment_id INTEGER,
                metric_name TEXT,  -- 'accuracy', 'auc', 'f1', 'precision', 'recall'
                metric_value FLOAT,
                subgroup TEXT,  -- NULL for overall, or demographic group
                split_type TEXT,  -- 'train', 'val', 'test'
                epoch INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Attention maps and visualizations
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_map_id START 1;
            CREATE TABLE IF NOT EXISTS attention_maps (
                map_id INTEGER PRIMARY KEY DEFAULT nextval('seq_map_id'),
                experiment_id INTEGER,
                wsi_id TEXT,
                file_path TEXT,
                prediction TEXT,
                ground_truth TEXT,
                confidence FLOAT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
                FOREIGN KEY (wsi_id) REFERENCES wsi_images(wsi_id)
            )
        """)
        
        # Create indexes for common queries
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for performance optimization."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_cases_mrn ON cases(mrn)",
            "CREATE INDEX IF NOT EXISTS idx_cases_recv_date ON cases(recv_date)",
            "CREATE INDEX IF NOT EXISTS idx_specimens_accession ON specimens(accession_no)",
            "CREATE INDEX IF NOT EXISTS idx_wsi_specimen ON wsi_images(specimen_id)",
            "CREATE INDEX IF NOT EXISTS idx_extractions_accession ON text_extractions(accession_no)",
            "CREATE INDEX IF NOT EXISTS idx_extractions_model ON text_extractions(model_name)",
            "CREATE INDEX IF NOT EXISTS idx_split_members_split ON data_split_members(split_id)",
            "CREATE INDEX IF NOT EXISTS idx_results_experiment ON model_results(experiment_id)",
        ]
        
        for idx_sql in indexes:
            try:
                self.conn.execute(idx_sql)
            except:
                pass  # Index may already exist
    
    def get_connection(self):
        """Get the database connection for direct queries."""
        return self.conn
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Query helper functions
class QueryHelper:
    """Helper class for common database queries."""
    
    def __init__(self, db: SAIFDatabase):
        self.db = db
        self.conn = db.conn
    
    def get_patient_summary(self) -> pd.DataFrame:
        """Get summary statistics of patients."""
        return self.conn.execute("""
            SELECT 
                COUNT(DISTINCT mrn) as total_patients,
                COUNT(DISTINCT CASE WHEN gender = 'M' THEN mrn END) as male_patients,
                COUNT(DISTINCT CASE WHEN gender = 'F' THEN mrn END) as female_patients,
                AVG(age) as avg_age,
                MIN(age) as min_age,
                MAX(age) as max_age
            FROM patients
        """).df()
    
    def get_case_summary(self) -> pd.DataFrame:
        """Get summary statistics of cases."""
        return self.conn.execute("""
            SELECT 
                COUNT(DISTINCT accession_no) as total_cases,
                COUNT(DISTINCT mrn) as unique_patients,
                MIN(recv_date) as earliest_case,
                MAX(recv_date) as latest_case,
                COUNT(DISTINCT YEAR(recv_date)) as years_span
            FROM cases
        """).df()
    
    def get_cases_by_patient(self, mrn: str) -> pd.DataFrame:
        """Get all cases for a specific patient."""
        return self.conn.execute("""
            SELECT * FROM cases WHERE mrn = ?
            ORDER BY recv_date DESC
        """, [mrn]).df()
    
    def get_extraction_coverage(self) -> pd.DataFrame:
        """Get coverage of text extractions by model."""
        return self.conn.execute("""
            SELECT 
                model_name,
                COUNT(DISTINCT accession_no) as cases_extracted,
                COUNT(*) as total_extractions,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM text_extractions
            GROUP BY model_name
        """).df()
    
    def search_cases(self, 
                     mrn: Optional[str] = None,
                     year_start: Optional[int] = None,
                     year_end: Optional[int] = None,
                     case_type: Optional[str] = None) -> pd.DataFrame:
        """Search cases with filters."""
        
        conditions = []
        params = []
        
        if mrn:
            conditions.append("mrn = ?")
            params.append(mrn)
        
        if year_start:
            conditions.append("YEAR(recv_date) >= ?")
            params.append(year_start)
        
        if year_end:
            conditions.append("YEAR(recv_date) <= ?")
            params.append(year_end)
        
        if case_type:
            conditions.append("case_type = ?")
            params.append(case_type)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT * FROM cases 
            WHERE {where_clause}
            ORDER BY recv_date DESC
        """
        
        return self.conn.execute(query, params).df()
