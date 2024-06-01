import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import joblib
import io

# Load the trained model
model = joblib.load('placement_chance_model.pkl')

def predict_placement(df):
    # Make predictions
    predictions = model.predict(df[['marks', 'class_days', 'family_income']])
    
    # Add predictions to DataFrame
    df['placement_chance'] = predictions
    
    return df

def main():
    st.title("Student Placement Prediction")

    # Sidebar for single prediction
    st.sidebar.header("Single Prediction")
    marks = st.sidebar.number_input("Marks", min_value=0, max_value=100, value=50)
    class_days = st.sidebar.number_input("Class Days", min_value=0, max_value=365, value=180)
    family_income = st.sidebar.number_input("Family Income", min_value=0, max_value=1000000, value=50000)

    if st.sidebar.button("Predict Single"):
        single_df = pd.DataFrame({
            'marks': [marks],
            'class_days': [class_days],
            'family_income': [family_income]
        })
        single_df = predict_placement(single_df)
        st.sidebar.write("Single Prediction:")
        st.sidebar.write(single_df)

    # File upload for batch predictions
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = predict_placement(df)
        st.write("Predictions:")
        st.write(df)

        # Convert DataFrame to Excel
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        # Download link
        st.download_button(
            label="Download Excel file",
            data=excel_buffer,
            file_name="placement_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
