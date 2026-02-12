# SAIF Case Viewer

A comprehensive data management and visualization system for pathology reports, demographics, and whole slide images in the SAIF project.

## Features

### 🔬 Case Viewer (Main Page)
- **Patient Management**: Browse cases by MRN (Medical Record Number)
- **Case Selection**: View individual cases by accession number
- **LLM Extraction Results**: Display AI-extracted structured data (diagnoses, specimens)
- **Customizable Layout**: Row-based grid system with horizontal/vertical arrangements
- **Specimen Display**: Radio button selection for viewing extracted specimens
- **Date Filtering**: Filter cases by year of receipt
- **Flexible Navigation**: Search by MRN/accession number or use slider for random browsing

### 📊 Data Dashboard (New!)
Interactive analytics dashboard with hierarchical data visualization:

**Patient Demographics**
- Age distribution histogram
- Gender pie chart
- Race/ethnicity distribution
- Age group analysis
- Searchable patient table

**Case Distribution**
- Cases over time (monthly timeline)
- Cases by year
- Specimens per case distribution
- Slides per case distribution

**Specimen Analysis**
- Organ distribution (Breast, Skin, etc.)
- Slides per specimen statistics
- Specimen details table

**Slide Overview**
- Stain type distribution
- Slides per patient analysis
- Search by barcode/MRN/accession
- Full slide metadata with HPC paths

### 🗄️ Database Management
- **DuckDB Backend**: High-performance analytical database
- **Hierarchical Data Model**: Patient → Case → Specimen → Slide
- **Comprehensive Import**: Single CSV import for all levels
- **JSON Metadata**: Flexible storage for additional fields
- **Query Helpers**: Pre-built queries for common operations

## Installation

### Prerequisites
- Python 3.10 or higher
- Conda environment manager
- Pathology report data in CSV format
- DuckDB for database management

### Setup

1. **Create conda environment**:
```bash
conda create -n case_viewer python=3.10
conda activate case_viewer
```

2. **Install dependencies**:
```bash
cd case_viewer
pip install -r requirements.txt
```

3. **Initialize database**:
```bash
python init_db.py
```

This creates the database schema with tables for patients, cases, specimens, slides, text extractions, and experiments.

4. **Import comprehensive data**:
```bash
python import_comprehensive.py
```

This imports hierarchical data from demo_demographics.csv (patient demographics, case info, specimen metadata, and slide-level data).

5. **Prepare your data**:

Required CSV structure for comprehensive import (each row = one slide):
- **Patient level**: `MRN`, `SELF_REPORTED_RACE`, `SEX`, `YEAROFBIRTH`, `AGE`, `curated_race`, `age_group`, `curated_sex`, `genetic_ancestry`
- **Case level**: `accession_no`, `recv_date`, `accession_type`, `primary_specimen_division`, `specimen_description`
- **Specimen level**: `specimen` (A/B/C/D), `specimen_organ`
- **Slide level**: `barcode` (unique ID), `minerva_path` (HPC file path), `slide_type`, `slide_stain`, `scan_date`, `stain`, `consult_stain`, `filename`, `block`, `slide`

## Usage

### Quick Start

Launch the viewer from the case_viewer directory:

```bash
conda activate case_viewer
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Navigation

The application now has multiple pages:

**Main Page (Case Viewer):**
- Patient selection by MRN
- Case selection by accession number
- LLM extraction results display
- Full pathology report with customizable layout
- Year filtering

**Data Dashboard Page:**
- Patient demographics visualizations
- Case distribution over time
- Specimen and organ analysis
- Slide-level statistics with search
- Interactive Plotly charts

### Workflow Example

**For Case Review:**
1. Launch app: `streamlit run app.py`
2. Select a year range using the filter
3. Choose a patient by MRN
4. Select a case by accession number
5. View LLM extraction results and full report side-by-side
6. Customize layout using row/column configuration

**For Data Analysis:**
1. Navigate to "Data Dashboard" page in sidebar
2. Explore patient demographics (age, gender, race)
3. Analyze case trends over time
4. Review specimen organ distribution
5. Search for specific slides by barcode or accession
6. Export data for further analysis

## Data Requirements

### Comprehensive Hierarchical CSV

The system now supports importing all hierarchical data from a single denormalized CSV where each row represents one slide with all parent-level metadata:

```csv
MRN,accession_no,specimen,barcode,specimen_organ,recv_date,minerva_path,SELF_REPORTED_RACE,SEX,AGE,...
P000001,ACC-2024-00001,A,SLIDE001,Breast,2024-06-27,/path/to/slide.tiff,Asian,Female,53,...
P000001,ACC-2024-00001,B,SLIDE002,Breast,2024-06-27,/path/to/slide.tiff,Asian,Female,53,...
P000002,ACC-2024-00002,A,SLIDE003,Lung,2024-07-15,/path/to/slide.tiff,White,Male,67,...
```

**Key Columns:**
- **Unique Identifiers**: 
  - `barcode` - Unique slide ID
  - `accession_no` - Unique case ID
  - `MRN` - Unique patient ID
  - `specimen` - Specimen letter (A, B, C, D) within case
  
- **HPC Integration**:
  - `minerva_path` - Full path to WSI file on HPC cluster
  - `filename` - UUID filename
  
- **Demographics** (Original):
  - `SELF_REPORTED_RACE` - Self-reported race/ethnicity
  - `SEX` - Gender
  - `YEAROFBIRTH` - Birth year
  
- **Demographics** (Curated):
  - `curated_race` - Standardized race categories
  - `age_group` - Age categories (Adolescents, Adults, etc.)
  - `curated_sex` - Standardized gender
  - `genetic_ancestry` - Genetic ancestry data

### Database Schema

The DuckDB database includes 10 tables:

1. **patients** - Patient demographics (MRN, age, gender, race, metadata JSON)
2. **cases** - Case information (accession_no, MRN FK, recv_date, report_text, metadata JSON)
3. **specimens** - Specimen details (specimen_id, accession_no FK, specimen_number, organ, metadata JSON)
4. **wsi_images** - Slide metadata (wsi_id/barcode, specimen_id FK, file_path, stain_type, metadata JSON)
5. **text_extractions** - LLM extraction results (extraction_id, accession_no FK, model_name, results JSON)
6. **data_splits** - ML dataset splits (split_id, name, description, created_date)
7. **data_split_members** - Split membership (split_id FK, accession_no FK, subset, fold)
8. **experiments** - ML experiments (experiment_id, name, model_type, config JSON)
9. **model_results** - Model outputs (result_id, experiment_id FK, accession_no FK, predictions JSON)
10. **attention_maps** - Attention visualizations (map_id, result_id FK, file_path)

## Remote Sync with HPC

### Push Changes to Remote

After making local changes:

```bash
cd /path/to/data_viewer
git add case_viewer/
git commit -m "Description of changes"
git push origin case_viewer_dev
```

### Pull Updates from Remote

To get the latest changes from HPC or collaborators:

```bash
cd /path/to/data_viewer
git pull origin case_viewer_dev
```

### Check Status

```bash
git status
git log --oneline -5
```

## Development

### Project Structure

```
case_viewer/
├── app.py                          # Main Streamlit application with quick stats
├── pages/
│   └── 1_Data_Dashboard.py        # NEW: Analytics dashboard
├── viewers/
│   └── case_viewer.py             # Case viewer with LLM extraction display
├── db/
│   ├── __init__.py
│   ├── database.py                # DuckDB schema and connection
│   ├── importer.py                # Original text/extraction importers
│   └── hierarchical_importer.py   # NEW: Comprehensive CSV importer
├── demo_data/
│   ├── text/
│   │   └── demo_path_reports.csv  # Original pathology reports
│   ├── text_extraction/
│   │   └── GPT-5-mini/            # LLM extraction results
│   └── patient/
│       └── demo_demographic/
│           └── demo_demographics.csv  # NEW: Comprehensive hierarchical data
├── data/
│   └── saif.duckdb                # Database file (gitignored)
├── init_db.py                      # Database initialization script
├── import_comprehensive.py         # NEW: Import hierarchical data
├── requirements.txt
└── README.md
```

### Database Architecture

**Hierarchical Data Model:**
```
patients (MRN) 
    ↓
cases (accession_no, MRN FK)
    ↓
specimens (specimen_id, accession_no FK)
    ↓
wsi_images (wsi_id/barcode, specimen_id FK)
```

**ML Workflow Tables:**
```
data_splits → data_split_members → experiments → model_results → attention_maps
```

All tables include:
- Auto-incrementing primary keys (via DuckDB sequences)
- Foreign key constraints for data integrity
- JSON metadata columns for flexibility
- Timestamps (created_at, updated_at)

### Adding New Features

1. **New Dashboard Tab**: Add visualizations in `pages/1_Data_Dashboard.py`
2. **New Importer**: Create importer in `db/` directory
3. **Database Query**: Add helper methods to `db/database.py` QueryHelper class
4. **UI Module**: Add to `viewers/` for new display components
5. Test locally with sample data
6. Commit and push to `case_viewer_dev` branch

### Customization

**Database Queries**: Add custom queries to QueryHelper:
```python
class QueryHelper:
    def your_custom_query(self):
        return self.db.conn.execute("""
            SELECT ...
        """).fetchdf()
```

**Dashboard Charts**: Add new visualizations using Plotly:
```python
fig = px.histogram(df, x='column', title='Your Chart')
st.plotly_chart(fig, use_container_width=True)
```

**Import Custom Data**: Extend HierarchicalImporter for new data sources:
```python
class CustomImporter(HierarchicalImporter):
    def import_custom_data(self, file_path):
        # Your import logic
```

## TODO List

### High Priority

1. **🎨 Improve Dashboard Visualizations**
   - Change color schemes in demographics charts for better accessibility
   - Add color consistency across all visualizations
   - Implement custom color palettes (race/ethnicity, age groups)

2. **📊 Data Curation Interface**
   - Visualization of data splits and distributions
   - Interactive split creation (train/val/test)
   - Stratification options (by race, age, diagnosis, organ)
   - Export curated case lists for ML pipelines

3. **🔍 Advanced Search & Filtering**
   - Multi-field search (diagnosis keywords, organ, demographics)
   - Complex filter combinations
   - Save and load filter presets
   - Query builder interface

4. **📈 Experiment Tracking Dashboard**
   - Register ML experiments with configs
   - Import model results (metrics, predictions)
   - Performance visualization (ROC curves, confusion matrices)
   - Subgroup analysis (performance by race, age, etc.)
   - Attention map viewer integration

### Medium Priority

5. **🖼️ WSI Integration**
   - Import WSI metadata from scan directories
   - Extract magnification, stain type from file headers
   - Thumbnail generation and display
   - Link to minerva_path for HPC access

6. **📝 Text Extraction Enhancement**
   - Support multiple LLM models (compare GPT-5-mini vs others)
   - Extraction quality metrics
   - Manual correction interface
   - Extraction confidence scores

7. **👥 Patient Timeline View**
   - Chronological case history per patient
   - Longitudinal analysis
   - Multiple cases/specimens visualization
   - Treatment progression tracking

### Low Priority

8. **🔐 Data Versioning**
   - Track database schema versions
   - Data import audit logs
   - Rollback capabilities

9. **🌐 Multi-User Support**
   - User authentication
   - Role-based access control
   - Shared annotations/notes

10. **⚡ Performance Optimization**
    - Incremental data loading for large cohorts
    - Query optimization with indexes
    - Caching strategies for dashboard

## Troubleshooting

### "Database not found"
- Run `python init_db.py` to create database schema
- Check that `data/saif.duckdb` exists
- Verify you're in the `case_viewer` directory

### "CSV file not found"
- Verify the path to data files in `demo_data/`
- Check `demo_demographics.csv` exists for comprehensive import
- Update the path in import scripts if using custom location

### "Foreign key constraint violation"
- Delete and recreate database: `rm -f data/saif.duckdb && python init_db.py`
- Ensure import order: patients → cases → specimens → slides
- Use fresh import with `import_comprehensive.py`

### "Module not found: duckdb"
- Install dependencies: `pip install -r requirements.txt`
- Activate conda environment: `conda activate case_viewer`
- Verify installation: `python -c "import duckdb; print(duckdb.__version__)"`

### "Date parsing errors"
- Ensure `recv_date` column follows format: `YYYY-MM-DD HH:MM:SS`
- Check for missing or malformed dates in CSV
- Use pandas to preprocess dates if needed

### Port Already in Use
If port 8501 is busy:
```bash
streamlit run app.py --server.port 8502
```

### Dashboard Not Loading
- Check browser console for errors
- Clear Streamlit cache: `streamlit cache clear`
- Verify plotly is installed: `pip install plotly>=5.18.0`
- Restart app completely

## Performance Tips

- **Large Datasets (200k+ patients)**: 
  - Use year/organ filtering to limit displayed cases
  - Dashboard queries are optimized with DuckDB analytics
  - Search functionality filters at database level
- **Memory Management**: 
  - Close browser tabs when not in use
  - Clear Streamlit cache periodically
  - Use pagination for large result sets
- **Query Speed**: 
  - DuckDB provides columnar storage for fast aggregations
  - Indexes on MRN, accession_no, barcode for quick lookups
  - JSON metadata allows flexible queries without schema changes
- **Remote Access (HPC)**: 
  - Use SSH tunneling for secure access:
    ```bash
    ssh -L 8501:localhost:8501 user@hpc.server.edu
    # Then run: streamlit run app.py
    # Access at: http://localhost:8501
    ```
  - Consider using tmux/screen for persistent sessions
  - WSI files referenced by minerva_path (no need to move files)

## Recent Updates

### February 2026
- ✅ **Data Dashboard**: Interactive analytics with Plotly visualizations
- ✅ **Hierarchical Import**: Single CSV import for Patient→Case→Specimen→Slide
- ✅ **Demographics Support**: Original + curated race/age/gender fields
- ✅ **DuckDB Integration**: High-performance analytical database
- ✅ **WSI Metadata**: Slide-level data with HPC paths (minerva_path, barcode)
- ✅ **JSON Metadata**: Flexible storage for additional fields at all levels
- ✅ **Quick Stats**: Homepage displays patient/case/specimen/slide counts
- ✅ **Search Functionality**: Find slides by barcode, MRN, or accession number

### Previous Updates
- ✅ **LLM Extraction Display**: GPT-5-mini diagnostic extraction results
- ✅ **Customizable Layout**: Row-based grid system with 1-4 columns per row
- ✅ **Specimen Radio Buttons**: Accordion-style single selection
- ✅ **Filter System**: Show all data by default, narrow with filters
- ✅ **Model Name Detection**: Auto-extract from file path

## Citation

If you use this viewer, please cite the SAIF project.

## License

This project is part of the SAIF research project.

## Contact

For questions or issues, contact the research team.

---

**Version**: 2.0.0  
**Last Updated**: February 12, 2026  
**Database**: DuckDB 1.4.4  
**Python**: 3.10+  
**Key Dependencies**: streamlit, pandas, duckdb, plotly
