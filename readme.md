
# Energy Consumption Forecasting Using Machine Learning for Smart Grid Technology

## Project Overview

This project focuses on developing machine learning models to predict energy consumption (TSD) in smart grids. By analyzing historical data, such as energy demand, renewable generation, and settlement dates, it aims to enhance grid resource management and forecasting accuracy.

## Goals

- Predict future energy consumption in smart grids using historical data.
- Analyze the impact of renewable generation and settlement dates on energy usage.
- Optimize and combine machine learning models for reliable and accurate forecasts.

## Dataset

- **Source**: Electricity Consumption in the UK (2009-2022)
- **File Name**: `electricity_consumption_uk.csv`
- **Size**: 275,230 rows × 23 columns

## **Attributes:**

- **Numerical Attributes**: Settlement Period, National Demand (nd), Total System Demand (tsd), England Wales Demand, Embedded Wind Generation, Embedded Wind Capacity, Embedded Solar Generation, Embedded Solar Capacity, Non BM Storage, Pump Storage Pumping, IFA Flow, IFA2 Flow, BritNed Flow, Moyle Flow, East-West Flow, NEMO Flow, NSL Flow, ElecLink Flow, Scottish Transfer, Viking Flow, Is Holiday.
- **Categorical Attributes**: Settlement Date.

## Workflow

### 1. Data Preparation

- Loaded and analyzed the dataset using Pandas and Matplotlib.
- Conducted exploratory data analysis (EDA) to understand data patterns and anomalies.
- Adjusted, scaled, and cleaned key attributes like nd (National Demand) and tsd (Total System Demand).
- Created new features to enhance model inputs, such as `renewable_generation` and `Imp_Exp_flow` (Import and Export Flow).
- Handled missing values and outliers. Identified zero values in tsd and replaced outliers and zeros with rolling averages.

### 2. Visualizations

- Generated time series plots for various energy metrics such as nd, tsd, embedded_wind_generation, and embedded_solar_generation.
- Created yearly bar plots and scatter plots for specific flows, demand metrics, and renewable generation.
- Conducted seasonal and yearly comparisons of tsd, renewable_generation, and interconnector flows, including scatter plots with seasonal and holiday-specific patterns.

## ML Models

### 1. Linear Regression

#### 1.1 Outlier Detection and Special Analysis

- Developed methods for detecting and handling outliers in the dataset.
  - **Outlier Detection**:
    - Identified zero values in tsd.
    - Replaced outliers and zeros with rolling averages.
  - **Comparison of Attributes**:
    - Seasonal and yearly comparisons of tsd, renewable_generation, and interconnector flows.
    - Scatter plots with seasonal and holiday-specific patterns.
  - **Saving Results**:
    - Exported the cleaned dataset as an Excel file for further use.
  - **Special Analysis**:
    - Checked energy demand differences on holidays vs. non-holidays.
    - Assigned seasons and analyzed energy metrics per season.

### 1. Linear Regression
#### 1.2 Imports and Setup

- Essential libraries for preprocessing (pandas, MinMaxScaler), model training (LinearRegression, Ridge), hyperparameter tuning (GridSearchCV), and evaluation (mean_squared_error, r2_score).

#### 1.3 Data Loading and Preprocessing

- **Dataset Columns**: Retains relevant features like tsd, Imp_Exp_flow, renewable_generation, and settlement_date.
- **Conversion**: Encodes settlement_date as numeric for modeling.

#### 1.4 Feature Selection and Normalization

- Selected features (Imp_Exp_flow, renewable_generation, etc.) and normalized them using MinMaxScaler.

#### 1.5 Train-Test Split

- Split the dataset into training and testing subsets (80/20 split).

#### 1.6 Model Creation

- **Polynomial Features**: Used PolynomialFeatures(degree=2) for feature expansion.
- Fitted a Linear Regression model to the transformed data.

#### 1.7 Evaluation

- Calculated MSE and R² scores for training and test datasets.
- Visualized actual vs. predicted values.

#### 1.8 Hyperparameter Tuning

- Used GridSearchCV to find the best polynomial degree and Ridge regularization parameter (alpha).
- Fine-tuned the model using a pipeline with PolynomialFeatures and Ridge.

#### 1.9 Best Model Evaluation

- Evaluated the fine-tuned model with metrics (MSE, R²) and visualizations.
- Saved the best model using joblib.

#### 1.10 Results

- Identified the best polynomial degree and alpha value.
- Compared the performance of the fine-tuned model against the baseline.

#### 1.11 Saving the Model

- Saved the best-performing model to a file named `Correct_linearRegression.pkl`.

### 2. LSTM

#### 2.1 Setup and Imports

- Libraries for data preprocessing (pandas, MinMaxScaler), model building (tensorflow.keras), and evaluation (mean_absolute_error, mean_squared_error).

#### 2.2 Data Preprocessing

- Target and Features: Selected features include Imp_Exp_flow, renewable_generation, and england_wales_demand, with tsd as the target variable.
- Normalization: Applied Min-Max scaling to normalize the features and target.
- Train-Test Split: 80% of the data is used for training, and 20% for testing.

#### 2.3 Sequence Generation

- Optimized Function: Converts data into sequences for LSTM input. A sequence length of 48 is used, representing one day of 30-minute intervals.

#### 2.4 LSTM Model Construction

- Architecture:
  - Two LSTM layers with 50 units each and ReLU activation.
  - A dense output layer for predictions.
- Compilation: Uses Adam optimizer and Mean Squared Error (MSE) as the loss function.

#### 2.5 Training and Evaluation

- The model is trained for 3 epochs with a batch size of 32.
- Metrics used:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)

#### 2.6 Prediction and Visualization

- Predictions are made on the test set, and results are inverse-scaled to their original values.
- Actual vs. predicted values are plotted for comparison.

#### 2.7 Model Saving

- The trained LSTM model is saved to a file named `lstm_model.h5`.

### 3. XGBOOST

#### 3.1 Setup and Data Loading

- Libraries: xgboost, pandas, sklearn, joblib.
- Columns used include tsd, Imp_Exp_flow, renewable_generation, and settlement_date.
- Data is loaded and preprocessed for splitting.

#### 3.2 Data Splitting

- Split the dataset into training, testing, and hold-out sets based on date thresholds (`threshold_date_1` and `threshold_date_2`).
- Visualized the splits to ensure proper segmentation.

#### 3.3 Feature Engineering

- Converted settlement_date to numeric timestamps.
- Defined relevant features and target (tsd) for training and testing.

#### 3.4 Model Training

- An XGBoost Regressor is initialized with:
  - n_estimators: 1000
  - max_depth: 3
  - learning_rate: 0.01
  - early_stopping_rounds: 50
- Trained the model using the training set and evaluated it on the test set.

#### 3.5 Feature Importance

- Extracted and visualized feature importance as a bar chart to understand the impact of each feature.

#### 3.6 Predictions and Visualizations

- Made predictions on the test set.
- Visualized actual vs. predicted tsd values for the full test set and a two-week period.

#### 3.7 Error Metrics

- Computed:
  - MAPE (Mean Absolute Percentage Error)
  - RMSE (Root Mean Squared Error)

#### 3.8 Time Series Cross-Validation

- Utilized TimeSeriesSplit with a test size of 1 year and a gap of 48 time steps.
- Visualized train-test splits across multiple folds.

#### 3.9 Hyperparameter Tuning

- Used GridSearchCV to optimize max_depth, n_estimators, and subsample parameters.
- Evaluated the best model using cross-validation.

#### 3.10 Model Saving

- Saved the best XGBoost model as `Correct_xgb.pkl` for future use.

### 4. SARIMAX

#### 4.1 Setup and Data Loading

- Libraries: statsmodels, pandas, numpy, matplotlib, seaborn.
- Dataset loaded from an Excel file and settlement_date is set as the index.

#### 4.2 Preprocessing

- Data is resampled to daily frequency, and columns like month and year are added.
- Missing values in tsd are imputed using monthly averages.

#### 4.3 Feature Engineering

- Lag features are created (lag_day for 1-day lag and lag_year for 1-year lag).
- Differences between tsd and lagged values are calculated (difference_day, difference_year).

#### 4.4 Visualization

- Plots include:
  - Daily electricity demand (tsd).
  - Day-to-day and year-to-year differences.
  - Log-transformed tsd to stabilize variance.

#### 4.5 Stationarity Testing

- The Dickey-Fuller test is used to check stationarity.
- Rolling mean and standard deviation are visualized alongside the original series.

#### 4.6 Decomposition

- Time series is decomposed into trend, seasonality, and residuals using seasonal_decompose.

#### 4.7 Splitting the Data

- Data is split into training and testing sets based on a threshold date.

#### 4.8 SARIMA Modeling

- A custom function (`create_predict_analyse`) is defined to:
  - Fit a SARIMA model with specified (p, d, q) and seasonal (P, D, Q, m) parameters.
  - Plot diagnostics and confidence intervals.
  - Compute and visualize predictions.
  - Calculate MAPE (Mean Absolute Percentage Error).

#### 4.9 Model Comparison

- Multiple SARIMA configurations are tested with varying hyperparameters:
  - Model 0: (1, 0, 1) and (1, 0, 1, 12).
  - Model 1: (7, 1, 2) and (3, 1, 2, 12).
  - Model 2: (7, 1, 7) and (3, 1, 2, 12).
  - Model 3: Another configuration with fewer iterations but verbose diagnostics.
- Execution time and MAPE for each model are recorded and compared.

#### 4.10 Evaluation

- Predictions are visualized with confidence intervals.
- Models are evaluated based on MAPE and runtime.

### 5. Model Stacking

#### 5.1 Setup and Imports

- Libraries for machine learning (e.g., sklearn), time series analysis (e.g., tensorflow.keras for LSTM), and evaluation metrics.

#### 5.2 Loading Models

- Pre-trained models are loaded:
  - SARIMAX from a .pkl file.
  - XGBoost from another .pkl file.
  - LSTM from an .h5 file.

#### 5.3 Data Preparation

- Cleaned dataset is loaded and indexed by settlement_date.
- Features include Imp_Exp_flow and renewable_generation, with tsd as the target.
- Normalization is applied for LSTM using MinMaxScaler.

#### 5.4 Test Data Preparation

- Adds lag features (lag1, lag2, lag3) and temporal features (day_of_week, month, etc.) for XGBoost.
- Ensures that test data aligns with expected features for all models.

#### 5.5 Predictions

- SARIMAX: Predicts using `get_forecast`.
- XGBoost: Predicts directly using the prepared test features.
- LSTM: Generates sequences and predicts on the normalized test data.

#### 5.6 Aligning Predictions

- Ensures that predictions from all models (SARIMAX, XGBoost, LSTM) are of the same length.
- Trims predictions to the shortest available length.

#### 5.7 Stacking Model

- Combines predictions from all models as features for a meta-model.
- Uses LinearRegression as the meta-model to learn from these features and predicts the target (tsd).

#### 5.8 Evaluation

- Metrics calculated for the stacked model:
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - RMSE (Root Mean Squared Error)
- Visualizes actual vs. predicted values for the stacked model.

#### 5.9 Visualization

- A plot compares the stacked model's predicted values against the actual test data.

## UI

### 1. Helper Functions

- `input_to_df`: Converts user-provided feature inputs into a DataFrame that matches the expected model input.
- `load_model`: Loads the Linear Regression model from a .pkl file.
- `preprocess_data`: Prepares the uploaded dataset for prediction, converting settlement_date into numeric and aligning columns to match model expectations.

### 2. Streamlit Application Features

- **Title and Description**: Displays the app title and brief instructions for users.
- **File Uploaders**: Allows users to upload:
  - The Linear Regression model file (.pkl).
  - The dataset file (.xlsx).
- **Model and Dataset Handling**:
  - Loads the uploaded model and extracts the expected feature names.
  - Loads the dataset, displays its first few rows, and preprocesses it for compatibility with the model.
- **Prediction and Results**:
  - Uses the model to make predictions on the preprocessed dataset.
  - Displays the predictions alongside the original data in a table format.
  - Visualizes predictions as a line chart with settlement_date as the x-axis.
- **Downloading Predictions**: Provides an option to download the predictions as a .csv file.

### 3. Direct User Input for Prediction

- A form where users can manually input:
  - Imp_Exp_flow
  - renewable_generation
  - england_wales_demand
  - settlement_date
  - settlement_period
- Converts inputs into the required format and predicts TSD using the uploaded model.

### 4. Error Handling

- Validates the uploaded model and dataset.
- Provides user feedback on errors in preprocessing or predictions.

### 5. Visualization

- Includes an interactive line chart of predictions using Streamlit's charting tools.

## How to Use

### 1. Installation

- Install dependencies using `pip install -r requirements.txt`.

### 2. UI Features

- Upload a cleaned dataset.
- Choose prediction models (SARIMAX, XGBoost, LSTM, or Stacked).
- Visualize predictions with graphs.
- Download forecast results as a CSV file.

### 3. Direct Input

- Provide feature inputs like `Imp_Exp_flow`, `renewable_generation`, and `settlement_date` to get immediate predictions.

## Results

- Developed a reliable forecasting system with high accuracy.
- Enhanced interpretability of energy demand through visualizations.
- The MAPE and R² values listed here are derived from the future prediction dataset.

| Model                   | Metrics                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| Linear Regression       | Train MSE: 463734.24, Test MSE: 478985.81, Train R2: 0.9920, Test R2: 0.9918 |
| LSTM                    | MAPE: 12.05%                                                           |
| XGBoost (Before Tuning) | MAPE: 4.09%, RMSE: 1441.98                                             |
| XGBoost (After Tuning)  | MAPE: 2.82%, RMSE: 1091.40                                             |
| SARIMAX (Model 0)       | MAPE: 0.0076                                                          |
| SARIMAX (Model 1)       | MAPE: 0.0080                                                          |
| SARIMAX (Model 2)       | MAPE: 0.0532                                                          |
| SARIMAX (Model 3)       | MAPE: 0.0174                                                          |

## Future Work

- Improve the efficiency of the stacking model by exploring advanced meta-model techniques.
- Incorporate additional external data, such as weather and economic indicators, to enhance model accuracy.
- Expand the application to forecast energy consumption in real-time scenarios.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
