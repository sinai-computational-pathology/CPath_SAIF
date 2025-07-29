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
- Digitized dermatopathology slides from multiple sites
- Racially diverse patient population
- Sufficient tissue quality and coverage
- Availability of self-reported race and relevant clinical metadata
- Consent for research use

**Summary of the skin dataset by self-reported race compared with Mount Sinai healthcare system, New York City population, and public source (TCGA).**

| Self-reported Race | # Slides (%)      | Patients (%) | Health System Population (%) | NYC Population (%) | TCGA (%) |
|--------------------|------------------|--------------|-----------------------------|--------------------|----------|
| White              | 2,151 (40.8%)    | 39.3         | 43.1                        | 31.2               | 73.7     |
| Black              | 1,015 (19.3%)    | 19.0         | 21.7                        | 29.9               | 10.3     |
| Hispanic/Latino    | 868 (16.5%)      | 16.8         | 18.5                        | 21.0               | 8.5      |
| Asian              | 687 (13.1%)      | 15.7         | 10.3                        | 5.7                | 7.1      |
| Other              | 543 (10.3%)      | 9.3          | 6.4                         | 4.5                | 1.8      |
| **Total Number**   | 5,266            | 2,471        | 114,947                     | 8M                 | 23,276   |

## Workflow Overview

### Experiment 1: Baseline Race Prediction

- Train deep learning models to predict self-reported race from whole-slide images
- Evaluate baseline performance and identify confounding factors

### Experiment 2: Dataset Curation Strategies

- Apply three curation strategies to control for confounders (e.g., site, diagnosis, technical artifacts)
- Assess model performance for each strategy

### Experiment 3: Attention-Based Morphological Analysis

- Use attention-based mechanisms to identify race-associated morphological features
- Analyze attention maps to localize predictive regions (e.g., epidermis)
- Remove key regions and evaluate impact on model performance

### Slide Preprocessing

- Standardize image formats and magnification
- Quality control to filter out artifacts and ensure data integrity
- Prepare slides for model training and analysis

### Notes

- This codebase is designed for fairness and bias analysis in computational pathology
- Data privacy is a priority; public release is planned pending approval
- For questions or collaboration, please contact the repository maintainers
