import streamlit as st
import pandas as pd
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
import numpy as np
import folium
from streamlit_folium import folium_static
import googlemaps


def run():

    if st.button("Back to main"):
        st.session_state['current_page'] = 'main'
        st.experimental_rerun()

    # Address to location function
    # Take in address and output location
    api_key = 'AIzaSyDniLn_2nxEF5SnXMBc1yj9h6vLWGhkHfg'

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

    # Define offensive type and color
    offense_types = [
        'Drug/Narcotic Violations', 'Theft of Motor Vehicle Parts or Accessories',
        'Robbery', 'Destruction/Damage/Vandalism of Property',
        'Driving Under the Influence', 'Shoplifting', 'Theft From Motor Vehicle',
        'Wire Fraud', 'Theft From Building', 'Kidnapping/Abduction',
        'Counterfeiting/Forgery', 'All Other Larceny',
        'Pornography/Obscene Material', 'Extortion/Blackmail',
        'Motor Vehicle Theft', 'Sodomy', 'Stolen Property Offenses',
        'False Pretenses/Swindle/Confidence Game',
        'Credit Card/Automated Teller Machine Fraud', 'Embezzlement',
        'Identity Theft', 'Impersonation', 'Weapon Law Violations', 'Fondling',
        'Hacking/Computer Invasion', 'Drug Equipment Violations', 'Arson',
        'Human Trafficking, Commercial Sex Acts', 'Liquor Law Violations',
        'Bad Checks', 'Pocket-picking', 'Rape', 'Sexual Assault With An Object',
        'Purchasing Prostitution', 'Purse-snatching', 'Family Offenses, Nonviolent',
        'Theft From Coin-Operated Machine or Device',
        'Curfew/Loitering/Vagrancy Violations', 'Animal Cruelty',
        'Murder & Nonnegligent Manslaughter', 'Prostitution', 'Welfare Fraud',
        'Peeping Tom', 'Assisting or Promoting Prostitution', 'Drunkenness',
        'Justifiable Homicide', 'Incest', 'Statutory Rape', 'Bribery',
        'Operating/Promoting/Assisting Gambling', 'Betting/Wagering',
        'Negligent Manslaughter', 'Gambling Equipment Violation',
        'Trespass of Real Property', 'Simple Assault', 'Aggravated Assault',
        'Intimidation', 'Burglary/Breaking & Entering',
        'Human Trafficking, Involuntary Servitude'
    ]

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'lightblue', 'lightgreen',
              'gray', 'darkred', 'darkblue', 'darkgreen', 'darkpurple', 'cadetblue', 'lightgray',
              'black', 'beige', 'lightred', 'lightyellow', 'darkorange', 'midnightblue', 'lime',
              'lightpink', 'lavender', 'cyan', 'magenta', 'lightgreen', 'chocolate', 'gold', 'tan',
              'silver', 'peru', 'navy', 'maroon', 'crimson', 'olive', 'yellowgreen', 'indigo', 'ivory',
              'dodgerblue', 'fuchsia', 'coral', 'aqua', 'brown', 'khaki', 'orchid', 'yellow', 'salmon',
              'chartreuse', 'limegreen', 'slategray', 'darkkhaki', 'teal', 'deeppink', 'turquoise',
              'goldenrod', 'mediumseagreen', 'mediumblue', 'mediumvioletred', 'saddlebrown', 'firebrick',
              'mediumorchid', 'sienna', 'forestgreen', 'slateblue']

    offense_colors = dict(zip(offense_types, colors))

    df_split_1 = pd.read_csv("data/split_file_1.csv")
    df_split_2 = pd.read_csv("data/split_file_2.csv")
    df_split_3 = pd.read_csv("data/split_file_3.csv")

    # df_crime = pd.read_csv("data/crime_data_cleaned.csv")
    df_crime = pd.concat([df_split_1, df_split_2, df_split_3])
    df_crime = df_crime[~df_crime.index.duplicated(keep='first')]

    df_crime['Offense Start DateTime'] = pd.to_datetime(
        df_crime['Offense Start DateTime'])
    df_crime['Offense End DateTime'] = df_crime['Offense End DateTime'].combine_first(
        df_crime['Offense Start DateTime'] + pd.Timedelta(hours=1))

    # Define the function to find the optimal number of clusters

    def find_optimal_clusters(data, max_clusters):
        wcss = []

        for n in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, n_init=10, random_state=0).fit(
                data)  # Set n_init explicitly
            wcss.append(kmeans.inertia_)

        differences = np.diff(wcss)
        optimal_clusters = np.argmin(differences) + 2
        return optimal_clusters

    # Define the function to generate the crime map

    def generate_crime_map(offense_type, start_date, end_date, start_hour, end_hour, map_center):
        start_date = pd.to_datetime(start_date).normalize()
        end_date = pd.to_datetime(end_date).normalize(
        ) + pd.Timedelta(days=1)  # Include the end date fully

        # Filter the dataframe for the offense type
        offense_mask = (df_crime['Offense'] == offense_type)

        # Filter dataframe for the selected date range
        date_mask = (
            (df_crime['Offense Start DateTime'] >= start_date) &
            (df_crime['Offense Start DateTime'] < end_date)
        )

        # Filter datafram for the selected hour range
        hour_mask = (
            (df_crime['Offense Start DateTime'].dt.hour >= start_hour) &
            (df_crime['Offense Start DateTime'].dt.hour <= end_hour)
        )

        # Combine filters for offense and date and hour
        combined_mask = offense_mask & date_mask & hour_mask

        filtered_df = df_crime[combined_mask]

        # Coordinates for clustering
        coords = filtered_df[["Latitude", "Longitude"]].values

        # Finding optimal number of clusters
        max_clusters = 10
        optimal_clusters = find_optimal_clusters(coords, max_clusters)

        kmeans = KMeans(n_clusters=optimal_clusters, n_init=10,
                        random_state=0).fit(coords)  # Set n_init explicitly
        filtered_df['Cluster'] = kmeans.labels_

        m = folium.Map(location=map_center, zoom_start=14)

        # Create a Marker Cluster
        marker_cluster = MarkerCluster().add_to(m)

        # Add user address pin point
        if map_center:
            custom_icon = folium.CustomIcon(
                icon_image='image/user_address_pin_point.jpg',
                icon_size=(35, 35)
            )
            folium.Marker(
                map_center,
                popup='Your Address',
                icon=custom_icon
            ).add_to(m)

        for _, row in filtered_df.iterrows():
            color = offense_colors.get(row["Offense"], "darkblue")

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=row["Offense"],
                icon=folium.Icon(color=color)
            ).add_to(marker_cluster)

        return m

    # Streamlit part
    # st.set_page_config(layout="wide", page_title="Crime Map Visualization App")
    st.title("Crime Map Visualization App")

    # Sidebar for user inputs
    with st.sidebar:
        # Address inputs
        user_address = st.text_input("Enter an address for the map center")
        # Offense type inputes
        offense_type = st.selectbox("Select Offense Type", offense_types)

        # Display data time range
        st.write("Data from:\n1915-12-14 13:00:00 to 2023-10-12 01:17:00")
        st.write("1915-12-14 13:00:00 to 2023-10-12 01:17:00")

        # Date inputs
        start_date = st.date_input("Start Date", pd.to_datetime('2000-01-01'))
        end_date = st.date_input("End Date", pd.to_datetime('2023-01-01'))

        # Hour inputs as sliders or number input
        start_hour = st.number_input(
            'Start Hour', min_value=0, max_value=23, value=16, format="%d")
        end_hour = st.number_input(
            'End Hour', min_value=0, max_value=23, value=22, format="%d")

        generate_map = st.button("Generate Map")

    # display map
    if generate_map and user_address:
        lat, lon = get_location_by_address_google(user_address, api_key)
        if lat is not None and lon is not None:
            map_center = [lat, lon]
            # Generate the map with the specified parameters
            map_generated = generate_crime_map(
                offense_type, start_date, end_date, start_hour, end_hour, map_center)
            # Use `folium_static` to render the map with specific width and height
            folium_static(map_generated, width=900, height=600)
        else:
            st.error("Address not found. Please retype the address.")
