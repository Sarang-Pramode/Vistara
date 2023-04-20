import requests
import streamlit as st
import geopandas as gpd
import plotly.express as px
from streamlit_plotly_events import plotly_events # pip install streamlit-plotly-events


# Suggest a good filename for this page
st.set_page_config(page_title='NYC 2017 LiDAR Data Downloader', page_icon='🌲', layout='wide', initial_sidebar_state='auto')

# Set the base URL of the S3 bucket
base_url = 'https://s3.amazonaws.com/www.treefolio.org-2.0/Package_Generated/'

filename = '12227'
#'https://s3.amazonaws.com/www.treefolio.org-2.0/Package_Generated/12227/2017/'
# specify the object URL of the bucket folder
object_url = f"https://s3.amazonaws.com/www.treefolio.org-2.0/Package_Generated/{filename}/2017/LasClassified_{filename}/lasFile_Reconstructed_{filename}.las"

st.write(object_url)

st.markdown(f'<a href="{object_url}" target="_blank">Open URL in new tab</a>', unsafe_allow_html=True)


st.write('https://s3.amazonaws.com/www.treefolio.org-2.0/Package_Generated/12227/2017/LasClassified_12227/lasFile_Reconstructed_12227.las')
#Download the file
r = requests.get(object_url)

#Save the file
with open(f'lasFile_Reconstructed_{filename}.las', 'wb') as f:
    f.write(r.content)
    st.write('File downloaded')

st.markdown("<h1 style= align:center;>NYC 2017 LiDAR Data Downloader</h1>", unsafe_allow_html=True)


filepath = 'NYC_Topobathy2017_LIDAR_Index/NYC_Topobathy2017_LIDAR_Index.shp'
nyc_las_tiles = gpd.read_file(filepath)

#Convert geometry to WGS84
nyc_las_tiles = nyc_las_tiles.to_crs('EPSG:4326')

# Create Plotly figure
fig = px.choropleth_mapbox(
    nyc_las_tiles,
    geojson=nyc_las_tiles.geometry,
    locations=nyc_las_tiles.index,
    mapbox_style='carto-positron',
    center={'lat': 40.73, 'lon':-73.9},#center={'lat': 40.74949210762701, 'lon':-73.97236357852755},
    zoom=9,
    opacity=0.25)

fig.update_layout(showlegend=False, title='NYC 2017 LIDAR Tiles',
                margin=dict(l=0,r=0,b=0,t=0),
                paper_bgcolor="cadetblue")

Tile_SelectionMade = plotly_events(fig)


if Tile_SelectionMade:
    st.write(Tile_SelectionMade)

    single_row = nyc_las_tiles.iloc[Tile_SelectionMade[0]['pointIndex']]

    st.write(single_row)


