"""
SAIF Case Viewer - Main Application
A lightweight data viewer for pathology reports.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from viewers.case_viewer import CaseViewer
from db.database import SAIFDatabase


def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="SAIF Case Viewer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Title and description
    st.title("🔬 SAIF Case Viewer")
    st.markdown("""
    **Pathology Report Viewer**
    
    Browse and view pathology reports by patient (MRN) and case (Accession Number).
    """)
    
    # Quick stats if database exists
    db_path = Path(__file__).parent / "data" / "saif.duckdb"
    if db_path.exists():
        try:
            db = SAIFDatabase(str(db_path))
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                patient_count = db.conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
                st.metric("👥 Patients", f"{patient_count:,}")
            with col2:
                case_count = db.conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                st.metric("📋 Cases", f"{case_count:,}")
            with col3:
                specimen_count = db.conn.execute("SELECT COUNT(*) FROM specimens").fetchone()[0]
                st.metric("🔬 Specimens", f"{specimen_count:,}")
            with col4:
                slide_count = db.conn.execute("SELECT COUNT(*) FROM wsi_images").fetchone()[0]
                st.metric("🖼️  Slides", f"{slide_count:,}")
        except Exception as e:
            st.info("📊 View detailed analytics in the **Data Dashboard** page")
    
    st.divider()
    
    # Render the case viewer
    viewer = CaseViewer()
    viewer.render()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
    SAIF Case Viewer v1.0 | February 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
