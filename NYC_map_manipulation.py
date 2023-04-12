import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px


# import folium
# from streamlit_folium import st_folium, folium_static


#st.set_page_config(layout='wide')

filepath = '2010 Census Tracts/geo_export_139fc905-b132-4c03-84d5-ae9e70dded42.shp'

gdf = gpd.read_file(filepath)

# Create Plotly figure
fig = px.choropleth_mapbox(
    gdf,
    geojson=gdf.geometry,
    locations=gdf.index,
    mapbox_style='carto-positron',
    center={'lat': 40.7, 'lon': -73.9},
    zoom=4
)

# Display Plotly figure in Streamlit app
st.plotly_chart(fig)

