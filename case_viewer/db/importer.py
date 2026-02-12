"""
Data import utilities for SAIF database
Handles importing CSV, JSON, and other data formats
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

from db.database import SAIFDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataImporter:
    """Import various data types into SAIF database."""
    
    def __init__(self, db: SAIFDatabase):
        """Initialize importer with database connection.
        
        Args:
            db: SAIFDatabase instance
        """
        self.db = db
        self.conn = db.conn
    
    def import_pathology_reports(self, csv_path: str, 
                                 update_existing: bool = False) -> Dict[str, int]:
        """Import pathology reports from CSV.
        
        Args:
            csv_path: Path to CSV file with columns: MRN, accession_no, recv_date, report_text
            update_existing: Whether to update existing records
            
        Returns:
            Dictionary with import statistics
        """
        logger.info(f"Importing pathology reports from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['MRN', 'accession_no', 'recv_date', 'report_text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Normalize column names
        df = df.rename(columns={'MRN': 'mrn'})
        
        # Parse dates
        df['recv_date'] = pd.to_datetime(df['recv_date'])
        
        stats = {
            'total_rows': len(df),
            'patients_added': 0,
            'patients_updated': 0,
            'cases_added': 0,
            'cases_updated': 0,
            'errors': 0
        }
        
        # Import patients first
        unique_mrns = df['mrn'].unique()
        for mrn in unique_mrns:
            try:
                # Check if patient exists
                existing = self.conn.execute(
                    "SELECT mrn FROM patients WHERE mrn = ?", [mrn]
                ).fetchone()
                
                if not existing:
                    self.conn.execute(
                        "INSERT INTO patients (mrn) VALUES (?)", [mrn]
                    )
                    stats['patients_added'] += 1
                elif update_existing:
                    self.conn.execute(
                        "UPDATE patients SET updated_at = CURRENT_TIMESTAMP WHERE mrn = ?",
                        [mrn]
                    )
                    stats['patients_updated'] += 1
            except Exception as e:
                logger.error(f"Error importing patient {mrn}: {e}")
                stats['errors'] += 1
        
        # Import cases
        for _, row in df.iterrows():
            try:
                # Check if case exists
                existing = self.conn.execute(
                    "SELECT accession_no FROM cases WHERE accession_no = ?",
                    [row['accession_no']]
                ).fetchone()
                
                if not existing:
                    self.conn.execute("""
                        INSERT INTO cases (accession_no, mrn, recv_date, report_text)
                        VALUES (?, ?, ?, ?)
                    """, [
                        row['accession_no'],
                        row['mrn'],
                        row['recv_date'],
                        row['report_text']
                    ])
                    stats['cases_added'] += 1
                elif update_existing:
                    self.conn.execute("""
                        UPDATE cases 
                        SET mrn = ?, recv_date = ?, report_text = ?, 
                            updated_at = CURRENT_TIMESTAMP
                        WHERE accession_no = ?
                    """, [
                        row['mrn'],
                        row['recv_date'],
                        row['report_text'],
                        row['accession_no']
                    ])
                    stats['cases_updated'] += 1
            except Exception as e:
                logger.error(f"Error importing case {row['accession_no']}: {e}")
                stats['errors'] += 1
        
        logger.info(f"Import completed: {stats}")
        return stats
    
    def import_text_extractions(self, json_path: str, 
                               model_name: str,
                               update_existing: bool = False) -> Dict[str, int]:
        """Import text extraction results from JSON.
        
        Args:
            json_path: Path to JSON file with extraction results
            model_name: Name of the model used for extraction
            update_existing: Whether to update existing extractions
            
        Returns:
            Dictionary with import statistics
        """
        logger.info(f"Importing text extractions from {json_path} (model: {model_name})")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        stats = {
            'total_extractions': len(data),
            'added': 0,
            'updated': 0,
            'errors': 0
        }
        
        for accession_no, extraction_result in data.items():
            try:
                # Check if extraction already exists
                existing = self.conn.execute("""
                    SELECT extraction_id FROM text_extractions 
                    WHERE accession_no = ? AND model_name = ?
                """, [accession_no, model_name]).fetchone()
                
                # Determine status based on results
                status = 'success'
                if 'llm_response' in extraction_result:
                    specimens = extraction_result['llm_response'].get('specimens', [])
                    if not specimens:
                        status = 'partial'
                else:
                    status = 'failed'
                
                results_json = json.dumps(extraction_result)
                
                if not existing:
                    self.conn.execute("""
                        INSERT INTO text_extractions 
                        (accession_no, model_name, results, status, extraction_date)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, [accession_no, model_name, results_json, status])
                    stats['added'] += 1
                elif update_existing:
                    self.conn.execute("""
                        UPDATE text_extractions 
                        SET results = ?, status = ?, extraction_date = CURRENT_TIMESTAMP
                        WHERE accession_no = ? AND model_name = ?
                    """, [results_json, status, accession_no, model_name])
                    stats['updated'] += 1
                
            except Exception as e:
                logger.error(f"Error importing extraction for {accession_no}: {e}")
                stats['errors'] += 1
        
        logger.info(f"Extraction import completed: {stats}")
        return stats
    
    def import_specimens_from_extractions(self, 
                                         model_name: str = None,
                                         overwrite: bool = False) -> Dict[str, int]:
        """Import specimen data from text extraction results.
        
        Args:
            model_name: Specific model to use, or None for most recent
            overwrite: Whether to overwrite existing specimen records
            
        Returns:
            Dictionary with import statistics
        """
        logger.info("Importing specimens from text extractions")
        
        # Get extractions
        if model_name:
            query = """
                SELECT accession_no, results 
                FROM text_extractions 
                WHERE model_name = ? AND status = 'success'
            """
            extractions = self.conn.execute(query, [model_name]).fetchall()
        else:
            query = """
                SELECT accession_no, results 
                FROM text_extractions 
                WHERE status = 'success'
            """
            extractions = self.conn.execute(query).fetchall()
        
        stats = {
            'total_extractions': len(extractions),
            'specimens_added': 0,
            'specimens_skipped': 0,
            'errors': 0
        }
        
        for accession_no, results_json in extractions:
            try:
                results = json.loads(results_json)
                
                if 'llm_response' not in results or 'specimens' not in results['llm_response']:
                    continue
                
                specimens = results['llm_response']['specimens']
                
                for specimen in specimens:
                    specimen_num = specimen.get('Specimen ID', 'Unknown')
                    specimen_id = f"{accession_no}_{specimen_num}"
                    
                    # Check if specimen exists
                    existing = self.conn.execute(
                        "SELECT specimen_id FROM specimens WHERE specimen_id = ?",
                        [specimen_id]
                    ).fetchone()
                    
                    if existing and not overwrite:
                        stats['specimens_skipped'] += 1
                        continue
                    
                    if existing:
                        self.conn.execute("""
                            UPDATE specimens 
                            SET specimen_source = ?, specimen_organ = ?, diagnosis = ?
                            WHERE specimen_id = ?
                        """, [
                            specimen.get('Specimen Source'),
                            specimen.get('Specimen Organ'),
                            specimen.get('Specimen Diagnosis'),
                            specimen_id
                        ])
                    else:
                        self.conn.execute("""
                            INSERT INTO specimens 
                            (specimen_id, accession_no, specimen_number, 
                             specimen_source, specimen_organ, diagnosis)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, [
                            specimen_id,
                            accession_no,
                            specimen_num,
                            specimen.get('Specimen Source'),
                            specimen.get('Specimen Organ'),
                            specimen.get('Specimen Diagnosis')
                        ])
                        stats['specimens_added'] += 1
                
            except Exception as e:
                logger.error(f"Error importing specimens for {accession_no}: {e}")
                stats['errors'] += 1
        
        logger.info(f"Specimen import completed: {stats}")
        return stats
    
    def import_patient_demographics(self, csv_path: str,
                                   update_existing: bool = True) -> Dict[str, int]:
        """Import patient demographic data from CSV.
        
        Args:
            csv_path: Path to CSV with patient demographics
            update_existing: Whether to update existing patient records
            
        Returns:
            Dictionary with import statistics
        """
        logger.info(f"Importing patient demographics from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Expected columns: mrn, age, gender, race, ethnicity, etc.
        if 'mrn' not in df.columns and 'MRN' not in df.columns:
            raise ValueError("CSV must contain 'mrn' or 'MRN' column")
        
        df = df.rename(columns={'MRN': 'mrn'})
        
        stats = {
            'total_rows': len(df),
            'added': 0,
            'updated': 0,
            'errors': 0
        }
        
        for _, row in df.iterrows():
            try:
                mrn = row['mrn']
                
                # Check if patient exists
                existing = self.conn.execute(
                    "SELECT mrn FROM patients WHERE mrn = ?", [mrn]
                ).fetchone()
                
                # Build update/insert columns dynamically
                columns = []
                values = []
                
                for col in ['age', 'gender', 'race', 'ethnicity', 'diagnosis_date', 'diagnosis_description']:
                    if col in row and pd.notna(row[col]):
                        columns.append(col)
                        values.append(row[col])
                
                if existing and update_existing:
                    set_clause = ", ".join([f"{col} = ?" for col in columns])
                    set_clause += ", updated_at = CURRENT_TIMESTAMP"
                    
                    self.conn.execute(
                        f"UPDATE patients SET {set_clause} WHERE mrn = ?",
                        values + [mrn]
                    )
                    stats['updated'] += 1
                    
                elif not existing:
                    columns.append('mrn')
                    values.append(mrn)
                    
                    placeholders = ", ".join(["?" for _ in values])
                    cols_str = ", ".join(columns)
                    
                    self.conn.execute(
                        f"INSERT INTO patients ({cols_str}) VALUES ({placeholders})",
                        values
                    )
                    stats['added'] += 1
                    
            except Exception as e:
                logger.error(f"Error importing demographics for {row.get('mrn')}: {e}")
                stats['errors'] += 1
        
        logger.info(f"Demographics import completed: {stats}")
        return stats
