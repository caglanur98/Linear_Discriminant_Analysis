# LDA on fMRI Data — Stimulation vs Imagery Classification

This repository contains MATLAB code and results for a Linear Discriminant Analysis (LDA) classification of fMRI data using the SPM12 toolbox.
The analysis aims to distinguish brain activity patterns between stimulation and imagery conditions in the right BA2 region of the primary somatosensory cortex, following the paradigm described in Nierhaus et al. (2023).

## Overview

We applied LDA to beta images from 10 subjects, extracted for:

- Stimulation: StimFlutt, StimVibro, StimPress

- Imagery: ImagFlutt, ImagVibro, ImagPress

For each subject, data from these sub-conditions were combined into the broader “Stimulation” and “Imagery” categories.

## Analysis Steps:

1. ROI extraction — masked beta images to right BA2 voxels.

2. Dimensionality reduction — Principal Component Analysis (PCA) to first two components.

3. Assumption checks:

- Henze–Zirkler test for multivariate normality

- Box’s M test for equal covariance matrices

- Bonferroni correction for multiple comparisons

4. Classification — LDA with 5-fold cross-validation per subject.

5. Significance testing — permutation tests (1,000 shuffles) per subject.

6. Group-level inference — one-sample t-test on classification accuracy vs. 50% chance.

7. Visualization — decision boundaries between Stimulation and Imagery plotted for each subject.
