import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import streamlit as st

try:
    # Read the CSV file
    data = pd.read_csv('Data.csv')
    
    # Drop unnecessary column
    data = data.drop('v.id', axis=1, inplace=False)

    # Prepare features (X) and target variable (y)
    X = data.drop('current price', axis=1, inplace=False).values
    y = data['current price'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Linear regression model
    lr = LinearRegression()
    model = lr.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display metrics
    print("MSE:", mean_squared_error(y_pred, y_test))
    print("RMSE:", np.sqrt(mean_squared_error(y_pred, y_test)))
    print("MAPE:", mean_absolute_percentage_error(y_pred, y_test) * 100)
    print("R-Square:", r2_score(y_pred, y_test))
    
except FileNotFoundError as e:
    print(f"Error: {e}. Check if the file 'Data.csv' exists.")
except ValueError as e:
    print(f"Error: {e}. Data processing error, check for missing values or improper data types.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Streamlit app title
st.title("Predict your car value 🚗")

# Creating dropdowns and text inputs
st.header("Please fill the required details 📄")
years = st.selectbox('Age of Car', list(range(1, 11)))
rating = st.selectbox('NCAP Rating', list(range(1, 6)))
condition = st.selectbox('Current Condition', list(range(1, 11)))
economy = st.selectbox('Mileage', list(range(1, 25)))
top_speed = st.selectbox('Top Speed', list(range(140, 241, 20)))

on_road_price = st.text_input("Purchased Price (On-Road)")
on_road_price_now = st.text_input("Current On-Road Price")
km = st.text_input("KM Driven")
torque = st.text_input("Torque")
hp = st.text_input("BHP")

# Handling user input and prediction
if st.button("Submit"):
    try:
        # Ensure all inputs are convertible to integers
        features = [on_road_price, on_road_price_now, years, km, rating, condition, economy, top_speed, hp, torque]
        features_int = [int(item) for item in features]

        # Reshape input to 2D array
        features_np_array = np.array(features_int)
        features_2d = features_np_array.reshape(1, -1)

        # Make prediction
        current_price = model.predict(features_2d)
        
        # Display prediction
        st.write("Predicted current price: ", current_price)

    except ValueError as e:
        st.error(f"Invalid input. Please enter valid numbers for all inputs. Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
