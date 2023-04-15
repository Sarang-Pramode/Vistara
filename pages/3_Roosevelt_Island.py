import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

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
from shapely.geometry import Point

# Roosevelt Island Index = 1625 , 1980

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

def stackTiles_NotLimited(lat,lon, boxSize=1500, prefix ='NY_NewYorkCity/'):  # 'NY_FingerLakes_1_2020/' #
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

def readGeoJSON(filepath):
    with open(filepath) as f:
        features = json.load(f)["features"]                
    return features

def convertXY(x,y):
    #translate from .las CRS to geojson CRS (NAD 1983) (UTM Zone 18N (meters))
    transformer = Transformer.from_crs( "epsg:2263", "epsg:4326" ) 
    lat, lon = transformer.transform(x, y)
    return lat, lon

###############################################################################

st.set_page_config(
        page_title="Roosevelt Island",
)

st.markdown("<h1 style='text-align: center; color: white;'>Roosevelt Island</h1>", unsafe_allow_html=True)


filepath = '2010 Census Tracts/geo_export_139fc905-b132-4c03-84d5-ae9e70dded42.shp'
gdf = gpd.read_file(filepath)

# # Create a map of the area
# selected_points = [
#   {
#     "curveNumber": 0,
#     "pointNumber": 1625, # This is the index of the point in the dataframe
#     "pointIndex": 1625
#   }
# ]


# Create a map of the area
lidarArea = 'NY_NewYorkCity/'

# Get indexs 1625 and 1980 from gdf
rows_to_keep = [1625, 1980]
RI_gdf = gdf.loc[rows_to_keep]

#st.write(RI_gdf)

# Get the latitude and longitude of the single row's geometry
lat, lon = 40.75538269567443, -73.9560580956196 

st.write(f"Location : {lat,lon}")

# Print the latitude and longitude
print("Latitude:", lat)
print("Longitude:", lon)

st.write(f"Selected Area : {lidarArea}")

#Plot RI_gdf

# fig = px.choropleth_mapbox(
#         RI_gdf,
#         geojson=RI_gdf.geometry,
#         locations=RI_gdf.index,
#         mapbox_style='carto-positron',
#         center={'lat': lat, 'lon': lon},
#         zoom=12)

# fig.update_layout(height=400, width=600, showlegend=False,
#                 margin=dict(l=0,r=0,b=0,t=0),
#                 paper_bgcolor="Black"
# )

# st.plotly_chart(fig)

# Get the point cloud data
Raw_lidar_df = stackTiles_NotLimited(lat,lon,prefix=lidarArea)

# Write the number of points found

st.write(f"Totol Number of Points {len(Raw_lidar_df)}")

#make a 3d plot of the points
Raw_lidar_df = Raw_lidar_df[::2]

# Create 3D scatter plot using Plotly with classification-based colors
fig = px.scatter_3d(Raw_lidar_df, x='X', y='Y', z='Z', color='Z',
                    hover_data=['X', 'Y', 'Z', 'class'],
                    color_continuous_scale=px.colors.sequential.Viridis,
                    width=400, height=500)

fig.update_traces(marker=dict(size=1.2))
fig.update_layout(scene=dict(aspectmode='data'), showlegend=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)


st.plotly_chart(fig, use_container_width=True)


# st.write(RI_new_lidar_df)

# # Perform a spatial join between the two dataframes
# joined = gpd.sjoin(gpd.GeoDataFrame(RI_lidar_df, geometry=gpd.points_from_xy(RI_lidar_df.X, RI_lidar_df.Y)), RI_gdf, op='within')

# #st.write(joined)

# st.write(f"Totol Number of Points {len(RI_lidar_df)}")

# # Create 3D scatter plot using Plotly with classification-based colors
# fig = px.scatter_3d(RI_lidar_df, x='X', y='Y', z='Z', color='class',
#                     hover_data=['X', 'Y', 'Z', 'class'])

# fig.update_traces(marker=dict(size=1.2))
# fig.update_layout(scene=dict(aspectmode='data'))

# st.plotly_chart(fig)
# st.write("Point cloud data:", RI_lidar_df)

# # Add a dpwnload button to download the RI_lidar_df
# st.download_button(
#     label="Download data as CSV",
#     data=RI_lidar_df.to_csv(index=False),
#     file_name='Raw_lidarData.csv',
#     mime='text/csv',
# )

# # Save the point cloud data to the session state
# st.session_state['Extracted_Lidar_Data_of_RI'] = RI_lidar_df
