# Machine Learning for Pre-Radiotherapy Tooth Extraction Prediction

This repository contains the code used in the thesis:

**_Machine Learning for Pre-Radiotherapy Tooth Extraction Prediction_**

## Data availability

The clinical tooth-extraction dataset is **confidential** and therefore **not included**.

- As a result, the following **cannot be executed as-is**:
  - `predictions/`
  - `analysis/`
  - `OHE_mitigation/CAE.ipynb`

However, the following folders have used a publically available dataset. The dataset can be imported by importing the ucimlrepo library using python:

- **`OHE_mitigation/`** — experiments with **categorical kernels**  
- **`SVM_Ensembles/`** — SVM ensembles using **categorical kernel k₁**

# Repository outline

**First step:** run `preprocessing.ipynb`.  
This notebook creates **two cleaned datasets** for prediction—one where `y` is based on **DMean** and another where `y` is based on **DMax**. This facilitates the modeling workflow so you don’t need to repeat cleaning each time. It also includes basic data analysis: **category frequencies** for categorical features and **summary statistics** for numerical features, corresponding to those found in section 4 of the thesis.

## Analysis

- **`dataAnalysis.ipynb`**  
    Brief exploratory analysis showing how features relate to whether an extraction is necessary **for both DMean and DMax**, corresponding to those displayed in section 4 of the thesis. 

## Modeling 

- **`predictions folder`**  
  Contains code to run **LR**, **SVM**, and **RF** models.  
  - Update the directory paths to point to the appropriate dataset (DMean or DMax) for each run.  
  - The scripts:
    - Fit the models and generate **predictions**
    - Report **evaluation metrics**
    - Plot **feature importance**
    - Plot the **distribution of predicted probabilities**

## Methological experiments (Categorical Kernels + AEs)

- **`OHE Mitigation folder`**  
  Contains code for experiments with **categorical kernels** 
  - **`customkernels.py`**: defines the kernel methods.
  - **`kernel1Pred.ipynb`**: SVM predictions using **categorical kernel 1**.
  - **`kernel2Pred.ipynb`**: SVM predictions using **categorical kernel 2**.
  - **`CAE.ipynb`**: defines the Categorical Autoencoder and code to make predictions with SVM using CAE feature representations

- **`SVM Ensembles folder`**  
contains the code that compares how SVM ensemble eprfroms using categorical kernel k_1
  - **`customkernels.py`**: defines the kernel methods.
  - **`Ens_reg.ipynb`**: Defines RSM and RSVM models employing rbf and linear kernels
  - **`Ens_k1.ipynb`**: Defines RSM and RSVM models employing k1 kernel
