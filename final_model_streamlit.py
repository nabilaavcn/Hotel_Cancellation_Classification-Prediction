# import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

#Judul Utama
st.title('Hotel Booking Cancellation Prediction')
st.text('This web can be used to predict hotel booking cancellations')


# Menambahkan sidebar
st.sidebar.header("Please input your features")

def create_user_input():
    # Numerical Features
    previous_cancellations = st.sidebar.slider('Previous Cancellations', min_value=0, max_value=26, value=0)
    booking_changes = st.sidebar.slider('Booking Changes', min_value=0, max_value=17, value=0)
    days_in_waiting_list = st.sidebar.slider('Days in Waiting List', min_value=0, max_value=391, value=0)
    required_car_parking_spaces = st.sidebar.slider('Required Car Parking Spaces', min_value=0, max_value=8, value=0)
    total_of_special_requests = st.sidebar.slider('Total of Special Requests', min_value=0, max_value=5, value=0)
    
    # Categorical Features
    market_segment = st.sidebar.radio('Market Segment', ['Groups', 'Direct', 'Online TA', 'Offline TA/TO', 'Complementary', 'Corporate', 'Aviation', 'Undefined'])
    deposit_type = st.sidebar.radio('Deposit Type', ['Non Refund', 'No Deposit', 'Refundable'])
    customer_type = st.sidebar.radio('Customer Type', ['Transient', 'Transient-Party', 'Contract', 'Group'])
    reserved_room_type = st.sidebar.radio('Reserved Room Type', ['D', 'A', 'F', 'B', 'H', 'E', 'C', 'G', 'P', 'L'])
    
    # Creating a dictionary with user input
    user_data = {
        'previous_cancellations': previous_cancellations,
        'booking_changes': booking_changes,
        'days_in_waiting_list': days_in_waiting_list,
        'required_car_parking_spaces': required_car_parking_spaces,
        'total_of_special_requests': total_of_special_requests,
        'market_segment': market_segment,
        'deposit_type': deposit_type,
        'customer_type': customer_type,
        'reserved_room_type': reserved_room_type
    }
    
    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df



# Get customer data
data_customer = create_user_input()

# Membuat 2 kontainer
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

# Load model
with open(r'Model Final.sav', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict to data
kelas = model_loaded.predict(data_customer)
probability = model_loaded.predict_proba(data_customer)[0]  # Get the probabilities

# Menampilkan hasil prediksi

# Bagian kanan (col2)
with col2:
    st.subheader('Prediction Result')
    if kelas == 1:
        st.write('Class 1: This customer will cancel their booking')
    else:
        st.write('Class 2: This customer will not cancel their booking')
    
    # Displaying the probability of the customer buying
    st.write(f"Probability of Cancellation: {probability[1]:.2f}")  # Probability of class 1 (BUY)
