# CAMS Particulate Matter Prediction Using Machine Learning

This repository contains machine learning models for predicting PM2.5 and PM10 concentrations using Copernicus Atmosphere Monitoring Service (CAMS) data, with validation against EPA Air Quality System (AQS) ground-truth observations.

## Data Sources

### CAMS Reanalysis Data
**Citation:** Inness, A., Ades, M., Agustí-Panareda, A., Barré, J., Benedictow, A., Blechschmidt, A.-M., Dominguez, J. J., Engelen, R., Eskes, H., Flemming, J., Huijnen, V., Jones, L., Kipling, Z., Massart, S., Parrington, M., Peuch, V.-H., Razinger, M., Remy, S., Schulz, M., and Suttie, M.: The CAMS reanalysis of atmospheric composition, Atmos. Chem. Phys., 19, 3515–3556, https://doi.org/10.5194/acp-19-3515-2019, 2019.

**Data Description:** CAMS provides global atmospheric composition reanalysis products including aerosol and trace gas concentrations. This project uses CAMS data from 2022-2024 covering the southeastern United States (30-36°N, 80-86°W).

### EPA Air Quality System (AQS)
**Data Source:** U.S. Environmental Protection Agency Air Quality System (AQS)  
**Website:** https://www.epa.gov/aqs

**Data Description:** EPA AQS provides ground-based PM2.5 measurements from monitoring stations across the United States. Daily PM2.5 observations (Parameter Code 88101) from 2022-2024 are used for comparison.

## Project Overview

This project develops and evaluates machine learning models to predict particulate matter concentrations (PM2.5 and PM10) using atmospheric composition and meteorological variables from CAMS reanalysis data.

### Study Region
- **Geographic Domain:** 30-36°N latitude, 80-86°W longitude (Southeastern United States)
- **Temporal Coverage:** Jan. 1, 2022 to Dec. 31, 2024
- **Spatial Resolution:** 0.75°x0.75° (CAMS resolution)
- **Temporal Resolution:** 3-hourly

## Notebooks

### 1. `run_models.ipynb` - Model Training

**Purpose:** Data processing, feature engineering, and training of machine learning models for PM prediction.

**Key Steps:**
1. **Data Loading & Merging**
   - Load CAMS GRIB files (surface and model level data)
   - Merge surface (GRIB1) and model level (GRIB2) datasets
   - Handle longitude coordinate transformations

2. **Unit Conversions**
   - Convert PM concentrations from kg/m³ to µg/m³
   - Convert trace gas mixing ratios to ppb using molar mass conversions
   - Calculate air density for aerosol mass conversions

3. **Feature Engineering**
   - Wind speed at 10m: `U10 = √(u10² + v10²)`
   - Total nitrogen oxides: `NOx = NO + NO2`
   - Total volatile organic compounds: `VOC = C2H6 + C3H8 + C5H8`
   - Relative humidity from temperature and dewpoint temperature

4. **Model Development**
   - **Features (14):** U10, C2H6, OH, C5H8, HNO3, NO2, NO, O3, C3H8, SO2, temperature, NOx, VOC, relative humidity
   - **Targets (2):** PM2.5, PM10
   - **Train/Test Split:** 70/30 temporal split
   - **Models:**
     - Random Forest
     - Random Forest with log-transform
     - XGBoost

5. **Hyperparameter Tuning**
   - GridSearchCV with TimeSeriesSplit (3-fold cross-validation)
   - Optimized for R² score
   - Random Forest parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf
   - XGBoost parameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_lambda

6. **Model Outputs**
   - `rf_best_model.pkl`: Best Random Forest model
   - `rf_best_model_log.pkl`: Best Random Forest model (log-transformed)
   - `xgb_best_model_log.pkl`: Best XGBoost model
   - `test_data.pkl`: Test dataset for model evaluation
   - JSON files with optimal hyperparameters for each of the three models

### 2. `model_comparison.ipynb` - Model Evaluation & Validation

**Purpose:** Evaluation of trained models including performance metrics, feature importance analysis, model interpretability, and validation against EPA AQS observations.

**Key Analyses:**

1. **Model Performance Comparison**
   - Calculate RMSE, MAE, MSE, and R² for all models
   - Compare Random Forest, Random Forest (log), and XGBoost
   - Generate performance metrics table and visualization

2. **Feature Importance Analysis**
   - **Gini Importance:**
   - **SHAP Analysis:**

3. **SHAP Visualizations**
   - Summary plots showing feature impact on predictions
   - Bar plots ranking features by importance
   - Heatmaps showing feature interactions
   - Waterfall plots for individual prediction explanations

4. **Ground Truth Validation**
   - Load EPA AQS daily PM2.5 observations for 2022-2024
   - Filter AQS data to study region (30-36°N, 80-86°W)
   - Calculate regional daily averages for both model predictions and AQS observations
   - Temporal alignment of predictions with ground-truth measurements

5. **Validation Metrics**
   - Model predictions vs AQS observations
   - CAMS input vs AQS observations (baseline comparison)
   - RMSE, MAE, R², and Bias calculations
   - Scatter plots: predicted vs observed
   - Time series comparison of model, CAMS, and AQS

6. **Key Findings**
   - Random Forest outperforms XGBoost (R² = 0.677 vs 0.633 for PM2.5)
   - Log transformation provides negligible improvement
   - Analysis focuses on Random Forest model for PM2.5 prediction
   - Ground truth validation quantifies model performance against independent observations

## Requirements

```python
numpy
pandas
xarray
scikit-learn
xgboost
shap
seaborn
matplotlib
joblib