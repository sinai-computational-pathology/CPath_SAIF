# Project Overview

This repository contains code for investigating fairness and bias in computational pathology (CPath) using artificial intelligence (AI). The project explores whether deep learning models can predict self-reported race from digitized dermatopathology slides and identifies potential morphological shortcuts. Our goal is to understand and mitigate unintended demographic biases in AI models for pathology.

This repository accompanies the following publication:

- **Publication:** Accepted to the MICCAI Workshop on Fairness of AI in Medical Imaging (FAIMI), 2025.
- **arXiv Link:** TBA

## Abstract

Artificial Intelligence (AI) has demonstrated success in computational pathology (CPath) for disease detection, biomarker classification, and prognosis prediction. However, its potential to learn unintended demographic biases, particularly those related to social determinants of health, remains understudied. This study investigates whether deep learning models can predict self-reported race from digitized dermatopathology slides and identifies potential morphological shortcuts. Using a multisite dataset with a racially diverse population, we apply an attention-based mechanism to uncover race-associated morphological features. After evaluating three dataset curation strategies to control for confounding factors, the final experiment showed that White and Black demographic groups retained high prediction performance (AUC: 0.799, 0.762), while overall performance dropped to 0.663. Attention analysis revealed the epidermis as a key predictive feature, with significant performance declines when these regions were removed. These findings highlight the need for careful data curation and bias mitigation to ensure equitable AI deployment in pathology.

## Data Summary

### Data Availability Statement

The multisite dataset used in this project is currently private due to institutional and patient privacy constraints. We are working towards making the data publicly available in the future.

### Inclusion Criteria

Slides included in this study meet the following criteria:

- Digitized dermatopathology slides within the Mount Sinai Health System (MSHS)
- Available self-reported race in patient level

-To prevent data leakage, slides from the same patient are not assigned to both training and validation folds.

**Summary of the skin dataset by self-reported race compared with Mount Sinai healthcare system, New York City population, and public source (TCGA).**

| Self-reported Race | # Slides (%)      | Patients (%) | Health System Population (%) | NYC Population (%) | TCGA (%) |
|--------------------|------------------|--------------|-----------------------------|--------------------|----------|
| White              | 2,151 (40.8%)    | 39.3         | 43.1                        | 31.2               | 73.7     |
| Black              | 1,015 (19.3%)    | 19.0         | 21.7                        | 29.9               | 10.3     |
| Hispanic/Latino    | 868 (16.5%)      | 16.8         | 18.5                        | 21.0               | 8.5      |
| Asian              | 687 (13.1%)      | 15.7         | 10.3                        | 5.7                | 7.1      |
| Other              | 543 (10.3%)      | 9.3          | 6.4                         | 4.5                | 1.8      |
| **Total Number**   | 5,266            | 2,471        | 114,947                     | 8M                 | 23,276   |

### Patient Ratio and Total Number by Experiment

| Experiment | Patients Ratio (%) | Total Number of Patients | White (%) | Black (%) | Hispanic/Latino (%) | Asian (%) | Other (%) |
|------------|-------------------|-------------------------|-----------|-----------|---------------------|-----------|-----------|
| Exp1       | 100               | 2,471                   | 39.3      | 19.0      | 16.8                | 15.7      | 9.3       |
| Exp2       | 82                | 2,028                   | 37.5      | 19.8      | 17.3                | 15.1      | 10.2      |
| Exp3       | 65                | 1,607                   | 46.9      | 19.9      | 19.6                | 7.2       | 6.5       |

We generated three versions of the experimental dataset to systematically control for confounding factors:

- **Exp1 (Uncurated):** Included all available dermatopathology specimens and yielded the highest overall OvR AUC (0.702), with particularly strong performance in the Asian group (AUC = 0.795). This was attributed to a disproportionately high prevalence of hemorrhoid cases (61%) among Asian patients due to site-specific sampling biases (160 out of 312 Asian patients treated at one site).
- **Exp2 (Balanced Disease):** Mitigated disease-related confounding by rebalancing hemorrhoid cases and removing conditions disproportionately prevalent in certain groups (e.g., gangrene, sun damage-related diagnoses, melanoma, basal cell carcinoma, squamous cell carcinoma, actinic keratosis, seborrheic keratosis), resulting in 2,028 patients (W 37.5%, B 19.8%, H/L 17.3%, A 15.1%, O 10.2%). This adjustment led to a decline in overall OvR AUC (0.671), with the Asian group experiencing the largest drop (AUC: 0.795 → 0.724).
- **Exp3 (Strict ICD Code):** Further restricted the dataset to classical dermatopathology cases (ICD-10 codes L, C, D), fully removing hemorrhoids (ICD-10 K), and reducing the dataset to 1,607 patients (W 46.9%, B 19.9%, H/L 19.6%, A 7.2%, O 6.5%). This further reduced the overall OvR AUC to 0.663, with the Asian group showing the most pronounced decline (0.570), whereas the White group maintained consistently high performance (0.799).

## Methods

### Slide Preprocessing

We standardized and prepared whole-slide images for model training using the script [`/data/make_features.py`](./data/make_features.py). The preprocessing workflow included:

- **Format Standardization:** Converted slides to a consistent image format (TIFF) and standardized magnification to 20x.
- **Tiling and Patch Extraction:** Divided each slide into non-overlapping 224x224 pixel tiles, filtering out tiles with low tissue content.
- **Feature Extraction:** Used a pretrained encoder (e.g., UNI, Virchow, Gigapath, SP22M, ...) to extract feature vectors from each tile for downstream analysis.
- **Metadata Integration:** Linked extracted features to patient-level metadata for stratified experiments.

To run the slide preprocessing on a Linux system, use the following example command:

```bash
python /data/make_features/make_features.py --meta_data_csv <meta_data.csv> --encoder <encoder_name> --tilesize 224 --bsize 512 --workers 8
```

- Replace `<meta_data.csv>` with your metadata filename. An example toy metadata file can be found at [`/data/self_reported_race/skin/master_metadata.csv`](./data/self_reported_race/skin/master_metadata.csv).

The script outputs feature embeddings for each slide are saved under `<output_dir>/<encoder>/features/<slide_name>.pth`. Additionally, the corresponding tile coordinates are stored at `<output_dir>/<encoder>/coordinates/<slide_name>.csv`. 

Alternatively, you can use feature embeddings generated from other preprocessing pipelines. We also recommend the [Trident pipeline](https://github.com/mahmoodlab/trident) for efficient and scalable whole-slide image preprocessing and feature extraction.

