import requests
import streamlit as st
import geopandas as gpd
import plotly.express as px
from streamlit_plotly_events import plotly_events # pip install streamlit-plotly-events
import matplotlib.pyplot as plt


# Set the base URL of the S3 bucket
base_url = 'https://s3.amazonaws.com/www.treefolio.org-2.0/'

st.markdown("# TreeFolio NYC Tree Data Downloader")

# st.markdown('TreeFolio is a web application that allows users to visualize and analyze tree data in New York City. The application is built using Python, Streamlit, and Plotly. The data is stored in an Amazon S3 bucket. The following code downloads a file from the S3 bucket and displays it in the Streamlit app.')

col1, col2 = st.columns(2)

with col1:


    # Add a title to the app
    st.markdown("<h1 style='text-align: center; color: white;'>About</h1>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: \
                left; color: white;'> \
                A large number of cities have been conducting LiDAR surveys to better map their cities. \
                Most of these surveys are public but requires significant technical expertise to extract and analyze. \
                To aid in the analysis of the neighbourhoods through LiDAR, I built this app to showcase the capabilities of an open source python package \
                I call <span style='color: #08a308; font-weight:bold;'>TerraVide</span>. You can check it out <a href='https://pypi.org/project/TerraVide/' target='_blank' style='color: #5f87c7;'>here</a><p> \
                <br> \
                Vistara was built to showcase some capabilites of <span style='color: #08a308; font-weight:bold;'>TerraVide</span>. \
                The tile you see on the right represents the classification of over <span style= font-weight:bold;'>12 Million</span> raw X,Y,Z coordinates into  \
                <span style= font-weight:bold;'>Trees</span>,\
                <span style= font-weight:bold;'>Buildings</span>, \
                and <span style= font-weight:bold;'>Ground</span></p>"
                , unsafe_allow_html=True)

with col2:

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
        center={'lat': 40.64, 'lon':-73.7},#center={'lat': 40.74949210762701, 'lon':-73.97236357852755},
        zoom=9,
        opacity=0.25)

    fig.update_layout(showlegend=False, title='NYC 2017 LIDAR Tiles',
                    margin=dict(l=0,r=0,b=0,t=0),
                    paper_bgcolor="cadetblue"
    )

    Tile_SelectionMade = plotly_events(fig)


if Tile_SelectionMade:
    st.write(Tile_SelectionMade)

    single_row = nyc_las_tiles.iloc[Tile_SelectionMade[0]['pointIndex']]

    st.write(single_row)

# #Create a matplotlib figure
# fig, ax = plt.subplots(figsize=(10, 10))

# # Plot the GeoDataFrame on the figure with no axis and add name of tile on each polygon
# nyc_las_tiles.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
# # for idx, row in nyc_las_tiles.iterrows():
# #     ax.annotate(row['FILENAME'], xy=row['geometry'].centroid.coords[0], ha='center')
# # Set the title of the figure
# ax.set_title('NYC LIDAR Tiles')
# # Remove the axis
# ax.axis('off')
# # Display the plot in Streamlit
# st.pyplot(fig)


# # Set the public URL of the file
# file_url = 'https://s3.amazonaws.com/www.treefolio.org-2.0/test.txt'

# # Send an HTTP GET request to the file URL
# response = requests.get(file_url)

# # Check if the request was successful
# if response.status_code == 200:
#     # Write the contents of the file to a local file
#     with open('local-file-name.txt', 'wb') as f:
#         f.write(response.content)
#         print('File downloaded successfully')
# else:
#     # Print an error message
#     print('Failed to download file. Error code:', response.status_code)


