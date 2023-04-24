import requests
import streamlit as st
import geopandas as gpd
import plotly.express as px
from streamlit_plotly_events import plotly_events # pip install streamlit-plotly-events


# Suggest a good filename for this page
st.set_page_config(page_title='NYC 2017 LiDAR Data Downloader', page_icon='🌲', layout='wide', initial_sidebar_state='auto')

# Set the base URL of the S3 bucket
#base_url = 'https://s3.amazonaws.com/www.treefolio.org-2.0/Package_Generated/'

st.markdown("<h1 style= align:center;>NYC 2017 LiDAR Data Downloader</h1>", unsafe_allow_html=True)

#add a large gap between the title and the rest of the page
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h3 style= align:center;>Select a tile from the map below to download data we host.</h3>", unsafe_allow_html=True)


##########################################################################################################################################

# Read in the NYC 2017 LIDAR Index Shapefile
filepath = 'NYC_Topobathy2017_LIDAR_Index/NYC_Topobathy2017_LIDAR_Index.shp'
nyc_las_tiles = gpd.read_file(filepath)

#Convert geometry to WGS84
nyc_las_tiles = nyc_las_tiles.to_crs('EPSG:4326')

# Create Plotly figure with hover data
fig = px.choropleth_mapbox(
    nyc_las_tiles,
    geojson=nyc_las_tiles.geometry,
    hover_name='FILENAME',
    locations=nyc_las_tiles.index,
    mapbox_style='carto-positron',
    color_discrete_sequence=['#f0f0f0'],
    center={'lat': 40.73, 'lon':-73.9},#center={'lat': 40.74949210762701, 'lon':-73.97236357852755},
    zoom=9,
    opacity=0.25)

fig.update_layout(showlegend=False, title='NYC 2017 LIDAR Tiles',
                margin=dict(l=0,r=0,b=0,t=0),
                paper_bgcolor="cadetblue")

fig.update_traces(marker_line_width=1, marker_line_color='black')

fig.update_layout(clickmode='event+select')

# Get the plotly events
Tile_SelectionMade = plotly_events(fig)

st.markdown("<p style='text-align: \
                center; color: white;'> \
             Each sqaure is a tile spanning approximately 0.25 sqmiles and roughly contains 12 - 18 million points. </p> \
             <p style='text-align: \
             center; color: white;'> \
             The 2017 LidAR scan contains 1800 of these files with close to 30 billion points spanning the whole of New York City</p>", unsafe_allow_html=True)

if Tile_SelectionMade:
    #st.write(Tile_SelectionMade)

    single_row = nyc_las_tiles.iloc[Tile_SelectionMade[0]['pointIndex']]

    st.write(single_row)

    tileFile_ID = single_row['FILENAME'][:-4]

    #Add a warning message
    st.warning("Warning!: The files you are about to download are large and may take a while to download. Please be patient.")



    #st.write(tileFile_ID)
    RawTile_object_url = f"https://s3.amazonaws.com/www.treefolio.org-2.0/nyc-Raw-LiDAR-2017/{tileFile_ID}.zip"
    
    # specify the object URL of the bucket folder
    ClassifiedTile_object_url = f"https://s3.amazonaws.com/www.treefolio.org-2.0/nyc-processed-2017-data/{tileFile_ID}.zip"

    #st.write(object_url)

    #Downlaod the Raw Tile
    r = requests.head(RawTile_object_url)
    if r.status_code == 200:
        #st.write('File Exists')
        st.markdown(f'## <a href="{RawTile_object_url}" target="_blank">Click Here To Download Raw Lidar Tile</a>', unsafe_allow_html=True)
        st.markdown(f'## <a href="{ClassifiedTile_object_url}" target="_blank">Click Here To Download Tile Dataset</a>', unsafe_allow_html=True)
    else:
        st.write('File does not exist! Please select another tile.')

    #Check if the file exists
    # r = requests.head(object_url)
    # if r.status_code == 200:
    #     #st.write('File Exists')
    #     st.markdown(f'## <a href="{object_url}" target="_blank">Click Here To Download Raw Lidar Tile</a>', unsafe_allow_html=True)
    # else:
    #     st.write('File does not exist! Please select another tile.')

##########################################################################################################################################

#add a large gap between the title and the rest of the page
st.markdown("<br>", unsafe_allow_html=True)

#Add a Horizontal Rule
st.markdown("<hr>", unsafe_allow_html=True)

#Learn more about our dataset
st.markdown("<h3 style= align:center;>Learn more about our dataset</h3>", unsafe_allow_html=True)

st.markdown("<p style='text-align: \
                left; color: white;'> \
                In 2017, New York City released a LiDAR scan of the entire city split into square tiles which is accessed through an FTP server. </p> \
                <p style='text-align: \
                left; color: white;'> \
                We have made all the processed data available for download free of charge and is to be used under the Creative Commons Attribution-NoDerivatives license. \
                This means that the dataset can be shared and used for any purpose, as long as it is attributed to the original source and is not modified in any way. </p> \
                When you click on an area on the map you will be able to download the following files: \
                <ul style='text-align: \
                left; color: white;'> \
                <li>Classified LiDAR Tile - .las</li> \
                <li>Raw Lidar Tile - .las</li> \
                <li>Indiviual Tree Models - .json</li> \
                <li>Individual Tree Summarized Shading Metrics - .csv</li> \
                <li>Individual Tree Shade Data - .json</li> \
                <li>HyperParameter Tile MetaData - .csv</li> \
                </ul> "\
                
                , unsafe_allow_html=True)


#Add a link to the Creative Commons Attribution-NoDerivatives license. 
st.markdown("<p style='text-align: \
                left; color: white;'> \
                <a href='https://creativecommons.org/licenses/by-nd/4.0/' target='_blank'> \
                Creative Commons Attribution-NoDerivatives license</a></p>", unsafe_allow_html=True)
