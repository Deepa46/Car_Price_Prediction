import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders/scalers
model = joblib.load('C:/Project_Guvi/Capstone3/cardekho_price_prediction_model.pkl')
label_encoders = joblib.load('C:/Project_Guvi/Capstone3/label_encoder.pkl')
scalers = joblib.load('C:/Project_Guvi/Capstone3/scaler.pkl')

# Load the cleaned dataset
df_raw = pd.read_csv('C:/Project_Guvi/Capstone3/Cleaned_Car_Dataset_Raw.csv', low_memory=False)

# Features used for training
features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage',
            'seats', 'car_age']

st.title('Car Price Prediction')

# Extract unique values for dropdowns
unique_oems = df_raw['oem'].unique()
models_by_oem = {oem: df_raw[df_raw['oem'] == oem]['model'].unique() for oem in unique_oems}

# Sidebar for user inputs
with st.sidebar:
    st.header('Enter Car Details')
    oem = st.selectbox('Car Brand (OEM)', unique_oems)
    if oem:
        model_name = st.selectbox('Model', models_by_oem[oem])

    fuel_type = st.selectbox('Fuel Type', df_raw['ft'].unique())
    body_type = st.selectbox('Body Type', df_raw['bt'].unique())
    transmission = st.selectbox('Transmission Type', df_raw['transmission'].unique())
    modelYear = st.number_input('Model Year', min_value=1980, max_value=2024, step=1)
    variantName = st.selectbox('Variant', df_raw['variantName'].unique())
    City = st.selectbox('City', df_raw['City'].unique())
    mileage = st.number_input('Mileage (kmpl)', min_value=0.0)
    seats = st.number_input('Number of Seats', min_value=2, max_value=10, step=1)
    km = st.number_input('Kilometers Driven', min_value=0, max_value=500000, value=10000)
    ownerNo = st.number_input('Number of Owners', min_value=0, max_value=5, step=1)

    # Prediction button
    predict_button = st.button('Predict Price')

# Calculate car age
current_year = 2024
car_age = current_year - modelYear


# Function to encode categorical variables
def encode_categorical(value, column_name):
    if column_name in label_encoders:
        encoder = label_encoders[column_name]
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            st.error(f"Value '{value}' for '{column_name}' is not recognized. Please select a valid option.")
            st.stop()  # Stop execution if an invalid value is selected
    else:
        st.error(f"Encoder for '{column_name}' is not properly loaded. Please check the encoder file.")
        st.stop()


# Function to scale numerical variables
def scale_numerical(value, column_name):
    if column_name in scalers:
        scaler = scalers[column_name]
        # Convert the single value to a DataFrame with the appropriate column name
        value_df = pd.DataFrame({column_name: [value]})
        return scaler.transform(value_df)[0][0]
    else:
        st.error(f"Scaler for '{column_name}' is not properly loaded. Please check the scaler file.")
        st.stop()


# Encode categorical inputs
ft_encoded = encode_categorical(fuel_type, 'ft')
bt_encoded = encode_categorical(body_type, 'bt')
transmission_encoded = encode_categorical(transmission, 'transmission')
oem_encoded = encode_categorical(oem, 'oem')
model_encoded = encode_categorical(model_name, 'model')
variantName_encoded = encode_categorical(variantName, 'variantName')
City_encoded = encode_categorical(City, 'City')

# Scale numerical inputs
km_scaled = scale_numerical(km, 'km')
modelYear_scaled = scale_numerical(modelYear, 'modelYear')
ownerNo_scaled = scale_numerical(ownerNo, 'ownerNo')
mileage_scaled = scale_numerical(mileage, 'mileage')
seats_scaled = scale_numerical(seats, 'seats')

# Prepare input data for prediction using a DataFrame
input_data = pd.DataFrame([[ft_encoded, bt_encoded, km_scaled, transmission_encoded, ownerNo_scaled,
                            oem_encoded, model_encoded, modelYear_scaled, variantName_encoded, City_encoded,
                            mileage_scaled, seats_scaled, car_age]],
                          columns=features)


# Function to format price with commas
def format_price(price):
    return f"₹{price:,.2f}"


# Display prediction result in the main area
if predict_button:
    try:
        predicted_price = model.predict(input_data)
        formatted_price = format_price(predicted_price[0])

        # Display prediction result with improved formatting and background color
        st.markdown(
            f"""
            <div style="background-color: #fef7e5; padding: 25px; border-radius: 10px; border: 1px solid #ddd;">
                <h2 class="prediction-title">Predicted Car Price</h2>
                <h2 style="text-align: center; color: green;">{formatted_price}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
