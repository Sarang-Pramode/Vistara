import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events # pip install streamlit-plotly-events

#plotly events inside streamlit - https://github.com/null-jones/streamlit-plotly-events

# #st.set_page_config(layout='wide')

import os
from pyproj import Transformer
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import json
import laspy
import numpy as np
import pandas as pd

def convertLatLon(lat,lon,epsgNumber):
    transformer = Transformer.from_crs( "epsg:4326", "epsg:{}".format(epsgNumber) ) 
    x, y = transformer.transform(lat, lon)
    return x, y

def getLazFile(lazfilename):
    with laspy.open(lazfilename) as lz:
        las = lz.read()
        lidarPoints = np.array((las.X,las.Y,las.Z,las.intensity,las.classification, las.return_number, las.number_of_returns)).transpose()
        lidarDF = pd.DataFrame(lidarPoints)
        lidarDF.columns = ['X', 'Y', 'Z', 'intens', 'class', 'return_number', 'number_of_returns']
        lidarDF['X'] = lidarDF['X'] * lz.header.scales[0] + lz.header.offsets[0]
        lidarDF['Y'] = lidarDF['Y'] * lz.header.scales[1] + lz.header.offsets[1]
        lidarDF['Z'] = lidarDF['Z'] * lz.header.scales[2] + lz.header.offsets[2]
    return lidarDF

def stackTiles(lat,lon, boxSize=100, prefix ='NY_NewYorkCity/'):  # 'NY_FingerLakes_1_2020/' #
    '''
    
    Parameters:
        lat : latitude centerpoint in WGS 1984 (EPSG 4326)
        lon : longitude centerpoint in WGS 1984 (EPSG 4326)
        boxSize : crop dimensions in X & Y units of source data, typically meters
        prefix : S3 server directory name for public usgs lidar, for available servers: https://usgs.entwine.io/
        
    Returns:
        lidar_df : Pandas dataframe containing selection of point cloud retrieved from S3 bucket
        
    
    '''
    low, high = 0,0
    
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket('usgs-lidar-public')
    for obj in bucket.objects.filter(Prefix= prefix + 'ept.json'):
        key = obj.key
        body = obj.get()['Body']
        eptJson = json.load(body)
        epsgNumber = eptJson['srs']['horizontal']
        span = eptJson['span']
        [xmin,ymin,zmin,xmax,ymax,zmax] = eptJson['bounds']
      
    x,y = convertLatLon(lat,lon,epsgNumber)   
    locatorx = ( x - xmin ) / ( xmax - xmin ) 
    locatory = ( y - ymin ) / ( ymax - ymin )
    
    try:
        os.mkdir('laz_{}/'.format(prefix))
    except:
        pass
    
    # download highest level laz for entire extent
    if os.path.exists('laz_{}/0-0-0-0.laz'.format(prefix)) == False:
        lazfile = bucket.download_file(prefix + 'ept-data/0-0-0-0.laz','laz_{}/0-0-0-0.laz'.format(prefix))
    else: 
        pass
    
    lidar_df = getLazFile('laz_{}/0-0-0-0.laz'.format(prefix))
            
    for depth in range(1,10):
        binx = int( (locatorx * 2 ** ( depth ) ) // 1 ) 
        biny = int( (locatory * 2 ** ( depth ) ) // 1 ) 
        lazfile = prefix + 'ept-data/{}-{}-{}-'.format(depth,binx,biny)
        for obj in bucket.objects.filter(Prefix = lazfile ):
            key = obj.key
            lazfilename = key.split('/')[2]
            # download subsequent laz files and concat 
            if os.path.exists('laz_{}/{}'.format(prefix,lazfilename)) == False:
                lazfile = bucket.download_file(prefix + 'ept-data/'+lazfilename,'laz_{}/{}'.format(prefix,lazfilename))
            else: 
                pass
            lidar_df2 = getLazFile('laz_{}/{}'.format(prefix,lazfilename))
            if depth > 7:
                low = lidar_df2['Z'].mean() - lidar_df2['Z'].std()*4
                high = lidar_df2['Z'].mean() + lidar_df2['Z'].std()*8
            else:
                low = 0
                high = 1000
            lidar_df = pd.concat([lidar_df,lidar_df2])
            
    lidar_df = lidar_df[lidar_df['Z'] > low ]
    lidar_df = lidar_df[lidar_df['Z'] < high ]
    lidar_df = lidar_df[lidar_df['X'] <= x + boxSize/2 ]
    lidar_df = lidar_df[lidar_df['X'] >= x - boxSize/2 ]
    lidar_df = lidar_df[lidar_df['Y'] <= y + boxSize/2 ]
    lidar_df = lidar_df[lidar_df['Y'] >= y - boxSize/2 ]
    return lidar_df

###############################################################################


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
    zoom=10)

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

    boxSize = 100

    lidar_df = stackTiles(lat,lon,boxSize)

    st.write(f"Totol Number of Points {len(lidar_df)}")

    # Create 3D scatter plot using Plotly with classification-based colors
    fig = px.scatter_3d(lidar_df, x='X', y='Y', z='Z', color='class',
                        hover_data=['X', 'Y', 'Z', 'class'])

    fig.update_traces(marker=dict(size=1.2))
    fig.update_layout(scene=dict(aspectmode='data'))

    st.plotly_chart(fig)
    st.write("Point cloud data:", lidar_df)

    # Save the point cloud data to the session state
    st.session_state['Extracted_Lidar_Data'] = lidar_df
