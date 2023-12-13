from cgi import print_arguments
from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import googlemaps
from sklearn.neighbors import KernelDensity
import numpy as np


def run():

    if st.button("Back to main"):
        st.session_state['current_page'] = 'main'
        st.experimental_rerun()

    # import sklearn
    # print(sklearn.__version__)

    # kde = load('model/crime_density_model.joblib')
    kde = load('model/crime_density_model_10.joblib')
    # kde = load('model/crime_density_model_100.joblib')crime_probability_pediction_combined

    def get_location_by_address_google(address, api_key):
        gmaps = googlemaps.Client(key=api_key)
        try:
            geocode_result = gmaps.geocode(address)
            if geocode_result:
                latitude = geocode_result[0]["geometry"]["location"]["lat"]
                longitude = geocode_result[0]["geometry"]["location"]["lng"]
                return latitude, longitude
            else:
                return None, None
        except googlemaps.exceptions.ApiError as e:
            print(f"API error: {e}")
            return None, None

    date_string = '1977-01-01 00:00:00'
    min_date = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')

    def predict_crime_density(longitude, latitude, time):
        date_time = pd.to_datetime(time)
        total_hours = (date_time - min_date).total_seconds() / 3600
        log_density = kde.score_samples([[longitude, latitude, total_hours]])
        print(log_density)
        return np.exp(log_density)  # Convert log density to actual density

    # Streamlit layout
    st.title('Area Safety Comparision Tool')

    # Sidebar for user inputs
    with st.sidebar:
        st.write("## Enter Addresses")
        # API key for Google Maps - consider security implications of hardcoding API keys
        api_key = 'AIzaSyDniLn_2nxEF5SnXMBc1yj9h6vLWGhkHfg'

        # User inputs for addresses
        main_address = st.text_input(
            'Enter the Main Address:')
        additional_addresses = st.text_area(
            'Enter other addresses (one per line)')

        # User input for date and time
        st.write("## Enter Date for Prediction")
        Date_time = st.date_input("Select the Date", datetime.now())

        # Button for calculations
        calculate_button = st.button('Calculate Crime Densities')

    # Process inputs
    if calculate_button:
        # Date_time = '2022-01-01 12:00:00'

        if main_address and additional_addresses and api_key:
            # Process main address
            latitude_main, longitude_main = get_location_by_address_google(
                main_address, api_key)

            # st.write("latitude_main:", latitude_main,
            #          longitude_main)  # Debug print

            if latitude_main is not None and longitude_main is not None:
                # Get crime density for the main address
                main_density = predict_crime_density(
                    latitude_main, longitude_main, Date_time)

                # st.write("main_density:", main_density)  # Debug print

                # Initialize list to store densities
                densities = []
                densities.append((main_address, 1))

                addresses = additional_addresses.split('\n')

                # Process each additional address
                for addr in addresses:
                    lat, lon = get_location_by_address_google(addr, api_key)

                    # st.write("additional add:", lat, lon)  # Debug print

                    if lat is not None and lon is not None:
                        density = predict_crime_density(lat, lon, Date_time)

                        # st.write("density:", density)  # Debug print

                        if density is not None and main_density is not None and main_density != 0:
                            relative_density = density / main_density
                            densities.append((addr, relative_density))
                        else:
                            st.write(
                                f"Invalid density value for address: {addr}")

                # Create DataFrame for plotting
                df_densities = pd.DataFrame(
                    densities, columns=['Address', 'Relative Density'])

                df_densities = df_densities.dropna()  # drop na
                df_densities = df_densities[df_densities['Relative Density'].apply(
                    lambda x: x is not None and not np.isnan(x))]

                # Check if the DataFrame is not empty and print its content
                if not df_densities.empty:
                    # st.write("Data for plotting:", df_densities)  # Debug print

                    address_list = df_densities['Address'].tolist()
                    density_list = df_densities['Relative Density'].tolist()
                    print(address_list)
                    print(df_densities['Relative Density'])
                    # Plotting
                    plt.figure(figsize=(10, 6))
                    num_addresses = len(df_densities['Address'])
                    bar_width = 0.1

                    if num_addresses < 5:
                        bar_width = 0.2
                    try:
                        bars = plt.bar(address_list,
                                       density_list, width=bar_width)
                        # bars = plt.bar(address_list,
                        #                df_densities['Relative Density'], width=bar_width)
                        plt.ylabel('Relative Crime Density')
                        plt.xticks(rotation=45)

                        plt.title(
                            'The area is considered safer if its y value is lower than that of the main address')
                        # add lables for each column
                        for bar in bars:
                            yval = bar.get_height()
                            yval = yval.item() if isinstance(yval, np.ndarray) else yval
                            plt.text(bar.get_x() + bar.get_width()/2, yval,
                                     round(yval, 2), va='bottom', ha='center')

                        plt.tight_layout()
                        st.pyplot(plt)
                    except TypeError as e:
                        st.error(f"An error occurred during plotting: {e}")
                else:
                    st.error("No data available for plotting.")

            else:
                st.error('Could not find the main address.')
        else:
            st.error('Please provide all required information.')
