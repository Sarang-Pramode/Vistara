import streamlit as st
import folium
import pandas as pd

from streamlit_folium import folium_static


st.set_page_config(page_title="Interactive Map with Streamlit", layout="wide")

st.title("Interactive Map with Streamlit")

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv")

# Create a basic Folium map
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Add markers to the map for each city
for index, row in df.iterrows():
    folium.Marker([row["lat"], row["lon"]], popup=row["name"]).add_to(m)

# Display the map using Streamlit
folium_static(m)
