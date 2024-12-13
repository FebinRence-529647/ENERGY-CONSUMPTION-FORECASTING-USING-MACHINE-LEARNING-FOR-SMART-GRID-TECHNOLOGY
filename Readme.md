# **Price Prediction of Used Cars in Canada**

## **Project Overview**

This project aims to build a machine learning model to predict the prices of used cars in Canada based on various attributes such as year, mileage, vehicle type, drivetrain, and more. The project utilizes multiple machine learning algorithms, evaluates their performance, and identifies the best-performing model for price prediction.

## **Goals**

- To help buyers and sellers of used cars estimate fair market prices.
- To analyze the impact of different vehicle attributes on pricing.
- To build a robust machine learning model for accurate predictions.

---

## **Dataset**

- **Source:** [Used Car listings for US & Canada](https://www.kaggle.com/datasets/rupeshraundal/marketcheck-automotive-data-us-canada?resource=download)
- **File Name:** `ca-dealers-used.csv`
- **Number of Rows:** 393,603
- **Number of Columns:** 21
- **Attributes:**
  - **Numerical:** Price, Miles, Year, Engine Size
  - **Categorical:** Make, Model, Body Type, Drivetrain, Transmission, Fuel Type
  - **Identifier:** VIN, ID, Stock Number
  - **Location:** City, State, ZIP Code

---

## **Workflow**

### 1. **Data Preparation**

- Loaded the dataset and performed exploratory data analysis (EDA).
- Identified numerical, categorical, and time-based attributes.
- Handled missing values and removed outliers.
- Encoded categorical variables using Label Encoding.

### 2. **Model Training**

- **Linear Regression:**
  - A simple regression model:
    ```python
    model = LinearRegression()
    model.fit(X_train, y_train)
    ```

- **XGBoost:**
  - Gradient-boosted decision tree:
    ```python
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    ```

- **Neural Network:**
  - Sequential model for deep learning:
    ```python
    model = Sequential([
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    ```

### 3. **Model Evaluation**

- Metrics for evaluation:
  - **Mean Absolute Error (MAE):**
    ```python
    mae = mean_absolute_error(y_test, y_pred)
    ```

  - **Mean Squared Error (MSE):**
    ```python
    mse = mean_squared_error(y_test, y_pred)
    ```

  - **R-squared (R²):**
    ```python
    r2 = r2_score(y_test, y_pred)
    ```

### 4. **Results**

| Model              | Dataset       | MAE        | MSE            | R² Score |
|--------------------|---------------|------------|----------------|------------|
| **Linear Regression** | Training      | 3,687.79   | 24,386,615.18  | 0.681      |
|                    | Test          | 3,651.17   | 23,885,529.70  | 0.682      |
| **XGBoost**        | Training      | 1,186.84   | 2,220,067.11   | 0.971      |
|                    | Test          | 1,430.25   | 3,920,632.97   | 0.948      |
| **Random Forest**  | Training      | 352.64     | 564,166.51     | 0.993      |
|                    | Test          | 859.32     | 2,435,353.20   | 0.968      |
| **Neural Network** | Training      | 2,358.54   | 11,480,009.00  | 0.850      |
|                    | Test          | 2,350.66   | 11,358,676.00  | 0.849      |

---

### 5. **Saving and Loading Models**

- Models are saved for reuse:
  ```python
  from joblib import dump, load
  dump(model, 'model_filename.joblib')
  ```

- To load the model:
  ```python
  model = load('model_filename.joblib')
  ```

### 6. **Interactive Prediction System**

- **Feature Options:**
  - Define lists for categorical options (e.g., `make_options`, `year_options`).

- **Model Selection:**
  - Dynamically load pre-trained models (Random Forest, XGBoost, Neural Network, Linear Regression).

- **Input Gathering:**
  - Collect user inputs interactively for features like:
    - Miles Driven
    - Year of Manufacture
    - Make (Brand), Model, Trim, etc.
  - Dynamically display options (e.g., models for a selected car make).

- **Feature Encoding:**
  - Encode categorical inputs using predefined dictionaries.

- **Prediction:**
  - Predict vehicle prices based on user input:
    ```python
    prediction = model.predict(encoded_features)
    ```

- **Loop for Continuous Use:**
  - Allow users to make multiple predictions or exit the program.

---

## **Installation**

### **Steps to Run the Project**

1. Download the Jupyter Notebook (`.ipynb`) file.
2. Open it in any Jupyter environment (e.g., Jupyter Notebook, JupyterLab, or Google Colab).
3. Run the cells step-by-step to execute the code and view the results.

---

## **Future Work**

- Add machine learning stacking techniques to improve prediction accuracy.
- Build a user interface (UI) for an interactive prediction system.

- Include more attributes like location-specific factors or condition of the vehicle.
- Explore hyperparameter tuning for Neural Network models.
- Deploy the model as a web application for real-time predictions.

---

## **Contact**

For any questions or collaboration, reach out to:

- **Name:** Febin
- **Email:** [[febinrence@gmail.com](mailto:your_email@example.com)]
