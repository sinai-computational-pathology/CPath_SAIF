"""
Hierarchical data importer for comprehensive CSV with patient/case/specimen/slide data
Handles denormalized CSV where each row is a slide with all parent-level metadata
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Set
import logging
import json
from db.database import SAIFDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalImporter:
    """Import hierarchical pathology data from denormalized CSV."""
    
    def __init__(self, db: SAIFDatabase):
        """Initialize with database connection.
        
        Args:
            db: SAIFDatabase instance
        """
        self.db = db
        self.conn = db.conn
        
    def import_comprehensive_data(self, csv_path: str, 
                                  update_existing: bool = True) -> Dict[str, int]:
        """Import patient, case, specimen, and slide data from hierarchical CSV.
        
        Expected CSV structure (each row = one slide with all parent metadata):
        - Patient level: MRN, SELF_REPORTED_RACE, SEX, YEAROFBIRTH, AGE, curated_race, 
                        age_group, curated_sex, genetic_ancestry
        - Case level: accession_no, recv_date, accession_type, primary_specimen_division,
                     specimen_description
        - Specimen level: specimen (A/B/C/D), specimen_organ
        - Slide level: barcode, minerva_path, slide_type, slide_stain, scan_date,
                      stain, consult_stain, filename, block, slide
        
        Args:
            csv_path: Path to comprehensive CSV file
            update_existing: Whether to update existing records
            
        Returns:
            Dictionary with import statistics
        """
        logger.info(f"Importing comprehensive data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['MRN', 'accession_no', 'specimen', 'barcode', 'minerva_path']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Parse dates
        if 'recv_date' in df.columns:
            df['recv_date'] = pd.to_datetime(df['recv_date'])
        if 'scan_date' in df.columns:
            df['scan_date'] = pd.to_datetime(df['scan_date'])
        
        stats = {
            'total_rows': len(df),
            'patients_added': 0,
            'patients_updated': 0,
            'cases_added': 0,
            'cases_updated': 0,
            'specimens_added': 0,
            'specimens_updated': 0,
            'slides_added': 0,
            'slides_updated': 0,
            'errors': 0
        }
        
        # Track unique entities to avoid re-processing
        processed_patients: Set[str] = set()
        processed_cases: Set[str] = set()
        processed_specimens: Set[str] = set()
        
        # Process each row (slide)
        for idx, row in df.iterrows():
            try:
                mrn = str(row['MRN'])
                accession_no = str(row['accession_no'])
                specimen_letter = str(row['specimen'])
                specimen_id = f"{accession_no}_{specimen_letter}"
                barcode = str(row['barcode'])
                
                # 1. Import/update patient (if not already processed)
                if mrn not in processed_patients:
                    patient_stats = self._import_patient(row, update_existing)
                    stats['patients_added'] += patient_stats['added']
                    stats['patients_updated'] += patient_stats['updated']
                    processed_patients.add(mrn)
                
                # 2. Import/update case (if not already processed)
                if accession_no not in processed_cases:
                    case_stats = self._import_case(row, update_existing)
                    stats['cases_added'] += case_stats['added']
                    stats['cases_updated'] += case_stats['updated']
                    processed_cases.add(accession_no)
                
                # 3. Import/update specimen (if not already processed)
                if specimen_id not in processed_specimens:
                    specimen_stats = self._import_specimen(row, specimen_id, update_existing)
                    stats['specimens_added'] += specimen_stats['added']
                    stats['specimens_updated'] += specimen_stats['updated']
                    processed_specimens.add(specimen_id)
                
                # 4. Import/update slide (always process - each row is unique slide)
                slide_stats = self._import_slide(row, specimen_id, update_existing)
                stats['slides_added'] += slide_stats['added']
                stats['slides_updated'] += slide_stats['updated']
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                stats['errors'] += 1
        
        logger.info(f"Import completed: {stats}")
        return stats
    
    def _import_patient(self, row: pd.Series, update: bool) -> Dict[str, int]:
        """Import patient demographics."""
        mrn = str(row['MRN'])
        
        # Check if exists
        existing = self.conn.execute(
            "SELECT mrn FROM patients WHERE mrn = ?", [mrn]
        ).fetchone()
        
        # Map demographics columns
        demo_mapping = {
            'race': 'SELF_REPORTED_RACE',
            'gender': 'SEX',
            'age': 'AGE'
        }
        
        # Build metadata JSON with curated fields
        metadata = {}
        for col in ['curated_race', 'age_group', 'curated_sex', 'genetic_ancestry', 
                   'SELF_REPORTED_RACE', 'YEAROFBIRTH']:
            if col in row and pd.notna(row[col]):
                metadata[col] = str(row[col])
        
        if existing and update:
            # Update
            self.conn.execute("""
                UPDATE patients 
                SET age = ?, gender = ?, race = ?, 
                    metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE mrn = ?
            """, [
                row.get('AGE') if 'AGE' in row and pd.notna(row['AGE']) else None,
                row.get('SEX') if 'SEX' in row and pd.notna(row['SEX']) else None,
                row.get('SELF_REPORTED_RACE') if 'SELF_REPORTED_RACE' in row and pd.notna(row['SELF_REPORTED_RACE']) else None,
                json.dumps(metadata) if metadata else None,
                mrn
            ])
            return {'added': 0, 'updated': 1}
        elif not existing:
            # Insert
            self.conn.execute("""
                INSERT INTO patients (mrn, age, gender, race, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, [
                mrn,
                row.get('AGE') if 'AGE' in row and pd.notna(row['AGE']) else None,
                row.get('SEX') if 'SEX' in row and pd.notna(row['SEX']) else None,
                row.get('SELF_REPORTED_RACE') if 'SELF_REPORTED_RACE' in row and pd.notna(row['SELF_REPORTED_RACE']) else None,
                json.dumps(metadata) if metadata else None
            ])
            return {'added': 1, 'updated': 0}
        
        return {'added': 0, 'updated': 0}
    
    def _import_case(self, row: pd.Series, update: bool) -> Dict[str, int]:
        """Import case data."""
        accession_no = str(row['accession_no'])
        mrn = str(row['MRN'])
        
        # Check if exists
        existing = self.conn.execute(
            "SELECT accession_no FROM cases WHERE accession_no = ?", [accession_no]
        ).fetchone()
        
        # Build metadata
        metadata = {}
        for col in ['accession_type', 'primary_specimen_division', 'specimen_description']:
            if col in row and pd.notna(row[col]):
                metadata[col] = str(row[col])
        
        if existing and update:
            self.conn.execute("""
                UPDATE cases 
                SET mrn = ?, recv_date = ?, metadata = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE accession_no = ?
            """, [
                mrn,
                row['recv_date'] if 'recv_date' in row else None,
                json.dumps(metadata) if metadata else None,
                accession_no
            ])
            return {'added': 0, 'updated': 1}
        elif not existing:
            self.conn.execute("""
                INSERT INTO cases (accession_no, mrn, recv_date, metadata)
                VALUES (?, ?, ?, ?)
            """, [
                accession_no,
                mrn,
                row['recv_date'] if 'recv_date' in row else None,
                json.dumps(metadata) if metadata else None
            ])
            return {'added': 1, 'updated': 0}
        
        return {'added': 0, 'updated': 0}
    
    def _import_specimen(self, row: pd.Series, specimen_id: str, update: bool) -> Dict[str, int]:
        """Import specimen data."""
        accession_no = str(row['accession_no'])
        specimen_letter = str(row['specimen'])
        
        # Check if exists
        existing = self.conn.execute(
            "SELECT specimen_id FROM specimens WHERE specimen_id = ?", [specimen_id]
        ).fetchone()
        
        # Build metadata
        metadata = {}
        for col in ['primary_specimen_division', 'specimen_description']:
            if col in row and pd.notna(row[col]):
                metadata[col] = str(row[col])
        
        if existing and update:
            self.conn.execute("""
                UPDATE specimens 
                SET specimen_organ = ?, metadata = ?
                WHERE specimen_id = ?
            """, [
                row.get('specimen_organ') if 'specimen_organ' in row and pd.notna(row['specimen_organ']) else None,
                json.dumps(metadata) if metadata else None,
                specimen_id
            ])
            return {'added': 0, 'updated': 1}
        elif not existing:
            self.conn.execute("""
                INSERT INTO specimens 
                (specimen_id, accession_no, specimen_number, specimen_organ, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, [
                specimen_id,
                accession_no,
                specimen_letter,
                row.get('specimen_organ') if 'specimen_organ' in row and pd.notna(row['specimen_organ']) else None,
                json.dumps(metadata) if metadata else None
            ])
            return {'added': 1, 'updated': 0}
        
        return {'added': 0, 'updated': 0}
    
    def _import_slide(self, row: pd.Series, specimen_id: str, update: bool) -> Dict[str, int]:
        """Import slide/WSI data."""
        barcode = str(row['barcode'])
        
        # Generate slide name from minerva_path
        slide_name = None
        if 'minerva_path' in row and pd.notna(row['minerva_path']):
            slide_name = Path(row['minerva_path']).name
        
        # Check if exists
        existing = self.conn.execute(
            "SELECT wsi_id FROM wsi_images WHERE wsi_id = ?", [barcode]
        ).fetchone()
        
        # Build metadata with slide-specific fields
        metadata = {}
        for col in ['filename', 'block', 'slide', 'consult_stain']:
            if col in row and pd.notna(row[col]):
                metadata[col] = str(row[col])
        
        if existing and update:
            self.conn.execute("""
                UPDATE wsi_images 
                SET file_path = ?, file_name = ?, stain_type = ?,
                    metadata = ?
                WHERE wsi_id = ?
            """, [
                row.get('minerva_path') if 'minerva_path' in row else None,
                slide_name,
                row.get('stain') if 'stain' in row and pd.notna(row['stain']) else None,
                json.dumps(metadata) if metadata else None,
                barcode
            ])
            return {'added': 0, 'updated': 1}
        elif not existing:
            self.conn.execute("""
                INSERT INTO wsi_images 
                (wsi_id, specimen_id, file_path, file_name, stain_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                barcode,
                specimen_id,
                row.get('minerva_path') if 'minerva_path' in row else None,
                slide_name,
                row.get('stain') if 'stain' in row and pd.notna(row['stain']) else None,
                json.dumps(metadata) if metadata else None
            ])
            return {'added': 1, 'updated': 0}
        
        return {'added': 0, 'updated': 0}
