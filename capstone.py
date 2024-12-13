import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Helper function to convert input features to DataFrame
def input_to_df(feature_inputs, expected_features):
    input_dict = {feature: [value] for feature, value in feature_inputs.items()}
    return pd.DataFrame(input_dict, columns=expected_features)

# Load the Linear Regression model
def load_model(uploaded_file):
    return pickle.load(uploaded_file)

# Preprocess input data
def preprocess_data(input_data, columns_to_keep, expected_features):
    # Convert settlement_date to a numeric format for model compatibility
    input_data['settlement_date_numeric'] = input_data['settlement_date'].astype('int64') // 10**9
    input_data = input_data[columns_to_keep + ['settlement_date_numeric']]  # Include the new numeric date column

    # Select only the features the model expects
    missing_features = [feature for feature in expected_features if feature not in input_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in the input data: {missing_features}")

    # Align the input data to the expected feature order
    input_data = input_data[expected_features]
    return input_data

# Main Streamlit app
def main():
    st.title("TSD Prediction Dashboard")
    st.write("Upload your Linear Regression model and dataset to predict TSD.")

    # File uploader for the model
    model_file = st.file_uploader("Upload Linear Regression Model (.pkl)", type=["pkl"])

    # File uploader for the dataset
    dataset_file = st.file_uploader("Upload your Dataset (.xlsx)", type=["xlsx"])

    columns_to_keep = ["tsd", "Imp_Exp_flow", "renewable_generation", "is_holiday", "england_wales_demand", "settlement_period"]

    if model_file and dataset_file:
        # Load the model
        st.write("Loading model...")
        model = load_model(model_file)
        st.success("Model loaded successfully!")

        # Extract expected feature names from the model
        expected_features = model.feature_names_in_

        # Load and display the dataset
        st.write("Loading dataset...")
        input_data = pd.read_excel(dataset_file, parse_dates=['settlement_date'])
        st.write("Uploaded Dataset:")
        st.dataframe(input_data.head())

        # Preprocess data for prediction
        st.write("Preprocessing data...")
        try:
            processed_data = preprocess_data(input_data, columns_to_keep, expected_features)
            st.success("Data preprocessing complete!")
        except ValueError as e:
            st.error(f"Error in preprocessing: {e}")
            return

        # Make predictions
        st.write("Making predictions...")
        predictions = model.predict(processed_data)
        input_data['TSD_Prediction'] = predictions

        # Display results
        st.write("### Predictions:")
        st.dataframe(input_data[['settlement_date', 'settlement_period', 'TSD_Prediction']])

        # Visualization
        st.write("### Visualization of Predictions:")
        chart_data = input_data[['settlement_date', 'TSD_Prediction']]
        chart_data = chart_data.set_index('settlement_date')  # Set settlement_date as the index
        st.line_chart(chart_data)

        # Option to download predictions
        st.write("### Download Predictions:")
        csv = input_data.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name="TSD_Predictions.csv", mime="text/csv")
    else:
        st.warning("Please upload both the model and the dataset to proceed.")

# Section for inputting values directly
    st.write("## Predict Future TSD by Inputting Feature Values")
    with st.form("prediction_form"):
        input_Imp_Exp_flow = st.number_input("Import Export Flow", format="%.2f")
        input_renewable_generation = st.number_input("Renewable Generation", format="%.2f")
        input_england_wales_demand = st.number_input("England Wales Demand", format="%.2f")
        input_date = st.date_input("Settlement Date")
        input_period = st.number_input("Settlement Period", format="%d", step=1)
        submit_button = st.form_submit_button("Predict TSD")


    if submit_button and model_file:
        # Convert input to DataFrame
        feature_inputs = {
            "Imp_Exp_flow": float(input_Imp_Exp_flow),
            "renewable_generation": float(input_renewable_generation),
            "england_wales_demand": float(input_england_wales_demand),
            "settlement_date_numeric": datetime.combine(input_date, datetime.min.time()).timestamp(),
            "settlement_period": int(input_period)
        }

        try:
            processed_data = input_to_df(feature_inputs, expected_features)
            #processed_data = processed_data.astype({key: 'float' for key in processed_data.columns if key != 'is_holiday'})
            predictions = model.predict(processed_data)
            st.write(f"### Predicted TSD: {predictions[0]}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    
    else:
        if submit_button:
            st.warning("Please upload the model to predict.")

# Run the app
if __name__ == "__main__":
    main()