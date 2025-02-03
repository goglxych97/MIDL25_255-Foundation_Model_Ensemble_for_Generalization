# Foundation Model Ensemble for Out-of-Distribution Generalization: Predicting Lymph Node Metastasis in Early Gastric Cancer Using Whole-Slide Imaging

**Status:** Under Review for MIDL 2025

This GitHub repository includes scripts for utilizing various foundation models and the entire pipeline used in our study. 
<br/>
  
## Repository Contents

- **model_weights/**: 
  - This folder contains the weights for various foundation models used in the study.

- **model_information.json**: 
  - This JSON file includes the settings for all foundation models used in the study.

- **model_usage_example.ipynb**: 
  - This notebook provides example code on how to use each foundation model as a feature extractor.
    
- **tiling.py**: 
  - This script contains functions for tiling Whole-Slide Images (WSIs) into predefined sizes.

- **training_example.ipynb**: 
  - This notebook is a training example for classification tasks. All models have been trained in the same method.
    
- **evaluate.py**: 
  - This script includes a function to evaluate models by averaging the top-100 patches from the heatmaps and applying the negative log likelihood function, for an assessment of model performance.
