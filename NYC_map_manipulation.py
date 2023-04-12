import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt

#st.set_page_config(layout='wide')

filepath = '2010 Census Tracts/geo_export_139fc905-b132-4c03-84d5-ae9e70dded42.shp'

@st.cache_data
def load_data(file_path):
    # Read in NYC census tracts shapefile
    gdf = gpd.read_file(file_path)

    return gdf

gdf = load_data(file_path=filepath)

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, alpha=0.5, edgecolor='black')
ax.set_axis_off()
st.pyplot(fig)
