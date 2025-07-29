Project Overview
This repository contains code for extracting and analyzing nucleus-level features from histopathology whole-slide images. The pipeline is designed for robust, scalable feature computation and spatial graph analysis.

Data Summary
Data Status: The dataset used in this project is currently private due to institutional and patient privacy constraints. We are actively working towards making the data publicly available in the future.
Data Table: [To be attached]
Inclusion Criteria
Only slides meeting the following criteria are included:
High-quality H&E-stained whole-slide images.
Sufficient tissue coverage and minimal artifacts.
Availability of relevant clinical metadata.
Consent for research use.
Workflow Overview
Experiment 1: Nucleus Feature Extraction
Load slide images using supported backends (openslide, tiffslide, PIL, or numpy).
Load nucleus centroids and contours from StarDist output.
Extract nucleus-level features:
Color statistics
Morphology metrics
Cytoplasm features
Haralick texture features
Gradient and intensity features
Fourier shape descriptors (FSD)
Save results in a structured table for downstream analysis.
Experiment 2: Spatial Graph Construction
Build Delaunay triangulation graphs using nucleus centroids.
Compute spatial relationships and neighbor-based statistics:
Mean, std, min, max distances
Cosine similarity of feature vectors
Neighbor feature aggregation
Integrate spatial features with nucleus-level data.
Experiment 3: Visualization and Quality Control
Generate visualizations of nuclei and their features.
Plot nuclei with bounding boxes and contours for inspection.
Save representative images for each slide and experiment group.
Slide Preprocessing
Support for multiple image loading methods to maximize compatibility.
Automatic magnification detection and adjustment.
Mask generation for nuclei and cytoplasm regions.
Efficient parallel processing for large-scale slides.
Quality control steps to filter out artifacts and ensure data integrity.
Notes
The code is modular and extensible for new feature types or analysis methods.
Data privacy is a priority; public release is planned pending approval.
For questions or collaboration, please contact the repository maintainers.
