import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events # pip install streamlit-plotly-events

#plotly events inside streamlit - https://github.com/null-jones/streamlit-plotly-events

# #st.set_page_config(layout='wide')


st.set_page_config(
        page_title="Data Extraction",
)

st.markdown("<h1 style='text-align: center; color: white;'>Project Vistara</h1>", unsafe_allow_html=True)

filepath = '../2010 Census Tracts/geo_export_139fc905-b132-4c03-84d5-ae9e70dded42.shp'

gdf = gpd.read_file(filepath)

# Create Plotly figure
fig = px.choropleth_mapbox(
    gdf,
    geojson=gdf.geometry,
    locations=gdf.index,
    mapbox_style='carto-positron',
    center={'lat': 40.64, 'lon':-73.7},#center={'lat': 40.74949210762701, 'lon':-73.97236357852755},
    zoom=9)

fig.update_layout(height=600, width=1000, showlegend=False,
                margin=dict(l=0,r=0,b=0,t=0),
                paper_bgcolor="Black"
)

selected_points = plotly_events(fig)


# # structure of selected_points
# [
#   {
#     "curveNumber": 0,
#     "pointNumber": 2150,
#     "pointIndex": 2150
#   }
# ]

if selected_points:

    st.write(selected_points)

    single_row = gdf.iloc[selected_points[0]['pointIndex']]

    # Get the latitude and longitude of the single row's geometry
    point = single_row.geometry.centroid
    lat, lon = point.y, point.x

    st.write(f"Location : {lat,lon}")

    # Print the latitude and longitude
    print("Latitude:", lat)
    print("Longitude:", lon)
