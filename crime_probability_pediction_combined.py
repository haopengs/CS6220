from calendar import month
from cgi import print_arguments
from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import googlemaps
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


def run():

    if st.button("Back to main"):
        st.session_state['current_page'] = 'main'
        st.experimental_rerun()

    df_split_1 = pd.read_csv("data/split_file_1.csv")
    df_split_2 = pd.read_csv("data/split_file_2.csv")
    df_split_3 = pd.read_csv("data/split_file_3.csv")

    # df_crime = pd.read_csv("data/crime_data_cleaned.csv")
    df_crime = pd.concat([df_split_1, df_split_2, df_split_3])
    df_zipcode_accuracy = pd.read_csv('data/model_results_by_zipcode.csv')
    df_month_accuracy = pd.read_csv('data/model_results_by_month.csv')

    unique_offenses = df_crime["Offense"].unique()

    label_encoder = LabelEncoder()
    df_crime['Offense_encoded'] = label_encoder.fit_transform(
        df_crime['Offense'])

    def get_location_by_address_google(address, api_key):
        gmaps = googlemaps.Client(key=api_key)
        try:
            geocode_result = gmaps.geocode(address)
            if geocode_result:
                latitude = geocode_result[0]["geometry"]["location"]["lat"]
                longitude = geocode_result[0]["geometry"]["location"]["lng"]
                zipcode = None

                address_components = geocode_result[0]["address_components"]
                for component in address_components:
                    if "postal_code" in component["types"]:
                        zipcode = component['long_name']
                        break
                return latitude, longitude, zipcode
            else:
                return None, None, None
        except googlemaps.exceptions.ApiError as e:
            print(f"API error: {e}")
            return None, None, None

    def predict_crime_probability(longitude, latitude, hour, offense_type, neural_network_model):
        # Convert hour into cyclical features
        hour_sin = np.sin(hour * (2. * np.pi / 24))
        hour_cos = np.cos(hour * (2. * np.pi / 24))

        # Create the input array
        input_data = np.array([[longitude, latitude, hour_sin, hour_cos]])

        input_data = input_data.reshape(1, 1, 4)

        # Get the predicted probabilities for all offense types
        predicted_probabilities = neural_network_model.predict(input_data)

        # Extract the index of the provided offense_type from the label encoder
        offense_index = label_encoder.transform([offense_type])[0]

        # Extract the predicted probability for the provided offense_type
        crime_probability = predicted_probabilities[0][offense_index]

        return crime_probability

    def load_crime_probability_model_by_zipcode(zipcode):
        model_path = f"model/crime_probability_models/crime_probability_model_{zipcode}.0"
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def load_crime_probability_model_by_month(month_input):
        model_path = f"model/crime_probability_models/crime_probability_model_{month_input}"
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    # Streamlit layout
    st.title('Crime Probability Prediction')

    # Sidebar for user inputs
    with st.sidebar:
        st.write("## Enter Details for Prediction")
        # API key for Google Maps - consider security implications of hardcoding API keys
        api_key = 'AIzaSyDniLn_2nxEF5SnXMBc1yj9h6vLWGhkHfg'

        # User inputs for address
        address = st.text_input('Enter Address')

        # User inputs for month
        month_input = st.text_input('Enter which month you want to predict')

        # Time input
        time = st.slider('Select Time (Hour of the Day)', 0, 23, 12)

        # Offense type selection
        offense_types_selected = st.multiselect(
            'Select Offense Types', unique_offenses)

        # Button for calculations
        predict_button = st.button('Predict Crime Probability')

    if predict_button and address and offense_types_selected:

        latitude, longitude, zipcode = get_location_by_address_google(
            address, api_key)
        # st.write(f"{longitude}: {latitude}")

        if zipcode:
            neural_network_model_by_zipcode = load_crime_probability_model_by_zipcode(
                zipcode)
            neural_network_model_by_month = load_crime_probability_model_by_month(
                month_input)

            if neural_network_model_by_zipcode and neural_network_model_by_month:
                if latitude is not None and longitude is not None:
                    # st.write(f"Predictions for {address} at {time}:00 hours")
                    st.markdown(
                        f"#### Predictions for {address} at time {time}:00: ")

                    # for offense_type in offense_types_selected:

                    #     probability = predict_crime_probability(
                    #         longitude, latitude, time, offense_type, neural_network_model)
                    #     st.write(
                    #         f"Probability of {offense_type} occurring at the given location and time: {probability * 100:.4f}% ")

                    probabilities_list = []

                    for offense_type in offense_types_selected:
                        final_probability = 0

                        probability_by_zipcode = predict_crime_probability(
                            longitude, latitude, time, offense_type, neural_network_model_by_zipcode)

                        probability_by_month = predict_crime_probability(
                            longitude, latitude, time, offense_type, neural_network_model_by_month)

                        accuracy_row_zipcode = df_zipcode_accuracy[df_zipcode_accuracy['Zipcode'] == int(
                            zipcode)]

                        accuracy_row_month = df_month_accuracy[df_month_accuracy['Month'] == int(
                            month_input)]

                        # weigh based on those two models' accuracy

                        # check if get the record
                        if not (accuracy_row_zipcode.empty or accuracy_row_month.empty):
                            # get accuracy
                            accuracy_value_zipcode = accuracy_row_zipcode['Accuracy'].iloc[0]
                            accuracy_value_month = accuracy_row_month['Accuracy'].iloc[0]
                            weight = accuracy_value_month + accuracy_value_zipcode

                            if (accuracy_value_zipcode - accuracy_value_month >= 0.1):
                                final_probability = probability_by_zipcode
                            else:
                                if (accuracy_value_zipcode != 0):
                                    final_probability = probability_by_zipcode * \
                                        (accuracy_value_zipcode / weight) + \
                                        probability_by_month * \
                                        (accuracy_value_month / weight)
                                else:
                                    final_probability = probability_by_month

                        else:
                            print(f"No data found, check zipcode or month")

                        probabilities_list.append(
                            f"Probability of {offense_type} occurring: {final_probability * 100:.4f}%")

                    # 以列表形式输出概率信息
                    st.markdown(
                        "\n".join(f"- {item}" for item in probabilities_list))

                else:
                    st.error("Could not find the specified address.")
            else:
                st.sidebar.error(
                    "Failed to load the model for the given zipcode.")
        else:
            st.sidebar.error("Zipcode not found for the given address.")
