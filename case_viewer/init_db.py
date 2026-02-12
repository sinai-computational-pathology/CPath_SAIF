"""
Initialize SAIF database with demo data
Run this script to set up the database and import demo data
"""

from pathlib import Path
from db.database import SAIFDatabase
from db.importer import DataImporter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_database():
    """Initialize database and import all demo data."""
    
    logger.info("="*60)
    logger.info("Starting SAIF Database Initialization")
    logger.info("="*60)
    
    # Create database
    db = SAIFDatabase(db_path="data/saif.duckdb")
    logger.info("✓ Database schema created")
    
    # Create importer
    importer = DataImporter(db)
    
    # Import pathology reports
    logger.info("\n" + "-"*60)
    logger.info("Importing pathology reports...")
    logger.info("-"*60)
    
    reports_path = "demo_data/text/demo_path_reports.csv"
    if Path(reports_path).exists():
        stats = importer.import_pathology_reports(reports_path)
        logger.info(f"✓ Pathology reports imported:")
        logger.info(f"  - Patients: {stats['patients_added']} added")
        logger.info(f"  - Cases: {stats['cases_added']} added")
        logger.info(f"  - Errors: {stats['errors']}")
    else:
        logger.warning(f"✗ Pathology reports file not found: {reports_path}")
    
    # Import text extractions
    logger.info("\n" + "-"*60)
    logger.info("Importing text extractions...")
    logger.info("-"*60)
    
    extraction_path = "demo_data/text_extraction/GPT-5-mini/demo_dx_extraction_gpt-5-mini.json"
    if Path(extraction_path).exists():
        stats = importer.import_text_extractions(
            extraction_path, 
            model_name="GPT-5-mini"
        )
        logger.info(f"✓ Text extractions imported:")
        logger.info(f"  - Extractions: {stats['added']} added")
        logger.info(f"  - Errors: {stats['errors']}")
    else:
        logger.warning(f"✗ Text extraction file not found: {extraction_path}")
    
    # Import specimens from extractions
    logger.info("\n" + "-"*60)
    logger.info("Importing specimens from extractions...")
    logger.info("-"*60)
    
    stats = importer.import_specimens_from_extractions()
    logger.info(f"✓ Specimens imported:")
    logger.info(f"  - Specimens: {stats['specimens_added']} added")
    logger.info(f"  - Errors: {stats['errors']}")
    
    # Display summary
    logger.info("\n" + "="*60)
    logger.info("Database Summary")
    logger.info("="*60)
    
    from db.database import QueryHelper
    query_helper = QueryHelper(db)
    
    patient_summary = query_helper.get_patient_summary()
    case_summary = query_helper.get_case_summary()
    extraction_summary = query_helper.get_extraction_coverage()
    
    logger.info("\nPatients:")
    logger.info(f"  Total: {patient_summary['total_patients'].values[0]}")
    
    logger.info("\nCases:")
    logger.info(f"  Total: {case_summary['total_cases'].values[0]}")
    logger.info(f"  Date range: {case_summary['earliest_case'].values[0]} to {case_summary['latest_case'].values[0]}")
    
    logger.info("\nText Extractions:")
    for _, row in extraction_summary.iterrows():
        logger.info(f"  {row['model_name']}: {row['cases_extracted']} cases ({row['successful']} successful)")
    
    logger.info("\n" + "="*60)
    logger.info("✓ Database initialization complete!")
    logger.info(f"✓ Database location: data/saif.duckdb")
    logger.info("="*60)
    
    db.close()


if __name__ == "__main__":
    initialize_database()
