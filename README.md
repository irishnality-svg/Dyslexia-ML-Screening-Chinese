# Dyslexia-ML-Screening-Chinese
Machine learning pipeline for identifying developmental dyslexia using Chinese character recognition.
# Identifying Children with Dyslexia via Machine Learning

This repository contains the data and R code necessary to reproduce the machine learning analyses, statistical tests, and visualizations presented in our manuscript.

##  Reproducibility & Pre-trained Models

To ensure **100% exact reproducibility** of the figures, tables, and statistical tests (e.g., DeLong Test, McNemar Test) reported in our paper, we have provided the final model objects as serialized `.rds` files alongside the clean dataset.

### Why use `.rds` files?
Machine learning pipelines—especially Neural Networks (like our MLP) and resampling techniques (like SMOTE)—introduce inherent computational randomness. While we have strictly set random seeds in our code (`set.seed(111)` and `tf$random$set_seed(111)`), the underlying multi-threading of TensorFlow/Keras and minor floating-point variations across different CPU/GPU architectures make it nearly impossible to achieve *pixel-perfect* replication when training from scratch on different local machines.

To eliminate this "irreproducibility" crisis, our `R code.Rmd` script is designed to directly load the pre-trained `.rds` objects. These lightweight files contain the exact predicted probabilities and optimized thresholds used in the paper, effectively freezing the results for transparent review.

### How to use this repository

#### Option A: Exact Replication (Recommended)
This is the default configuration of the script.
1. Ensure `dataset_anonymized.rds` and all model result files (e.g., `LR_results.rds`, `Vote_results.rds`) are in your working directory.
2. Open `R code.Rmd` and run the downstream code chunks. The script will automatically read the `.rds` files and instantly generate the ROC curves, SHAP summary plots, and metrics tables exactly as they appear in the manuscript.

#### Option B: Train from Scratch
If you wish to review our modeling methodology or train the models yourself:
1. The complete training functions (`run_lr_analysis`, `run_mlp_analysis`, `vote_optimize`, etc.) are fully preserved in the script.
2. You can uncomment the execution chunks (e.g., `run_mlp_analysis(...)`) to run the entire pipeline from scratch. 
*(Note: As explained above, due to Keras backend randomness, your resulting AUC/Accuracy may fluctuate slightly by ~0.01 compared to the original manuscript).*

## 📊 File Structure
* `R code.Rmd`: The main R Markdown file containing all functions, plotting scripts, and statistical tests.
* `dataset_anonymized.rds`: The cleaned, fully pre-processed dataset ready for machine learning.
* `*_results.rds`: Pre-trained model objects (LR, RF, XGBoost, MLP, and Vote Ensemble) containing probabilities and SHAP values.
