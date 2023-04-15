import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from shapely.ops import unary_union
from shapely.ops import cascaded_union



from scipy.spatial import cKDTree


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

from rtree import index

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

def stackTiles_NotLimited(lat,lon, boxSize=2000, prefix ='NY_NewYorkCity/'):  # 'NY_FingerLakes_1_2020/' #
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
    # lidar_df = lidar_df[lidar_df['X'] <= x + boxSize/2 ]
    # lidar_df = lidar_df[lidar_df['X'] >= x - boxSize/2 ]
    # lidar_df = lidar_df[lidar_df['Y'] <= y + boxSize/2 ]
    # lidar_df = lidar_df[lidar_df['Y'] >= y - boxSize/2 ]
    return lidar_df, epsgNumber

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
        page_title="Central Park",
)

st.markdown("<h1 style='text-align: center; color: white;'>Central Park</h1>", unsafe_allow_html=True)


st.markdown("## Analyzing the Central Park using LiDAR data")
filepath = '2010 Census Tracts/geo_export_139fc905-b132-4c03-84d5-ae9e70dded42.shp'
gdf = gpd.read_file(filepath)

#rows_to_keep = [1625, 1980]
Reduced_gdf = gdf.loc[[2150]] # central Park

#Get Bounding Box of Reduced_gdf
lon_min, lat_min, lon_max, lat_max = Reduced_gdf.total_bounds

# Get the center of the bounding box
lat = (lat_min + lat_max) / 2
lon = (lon_min + lon_max) / 2

# Print the latitude and longitude
print("Latitude:", lat)
print("Longitude:", lon)

#st.write(f"Selected Lidar Area : {lidarArea}")

#Plot RI_gdf
fig = px.choropleth_mapbox(
        Reduced_gdf,
        geojson=Reduced_gdf.geometry,
        locations=Reduced_gdf.index,
        mapbox_style='carto-positron',
        center={'lat': lat, 'lon':lon},
        zoom=12)

fig.update_layout(height=400, width=600, showlegend=False,
                margin=dict(l=0,r=0,b=0,t=0),
                paper_bgcolor="Black"
)

st.plotly_chart(fig)

# vertices = []

# # Check if the geometry is valid
# if not RI_gdf.geometry.is_valid.any():
#     st.write("Invalid geometry found, attempting to fix...")
#     # If the geometry is invalid, buffer it by a small distance to try to fix it
#     RI_gdf.geometry = RI_gdf.geometry.buffer(0.000001)

# # Iterate over the geometries in the GeoDataFrame
# for geom in RI_gdf.geometry:
#     #st.write(geom.geom_type)
#     # If the geometry is a Polygon, get its vertices
#     if geom.geom_type == 'Polygon':
#         vertices.extend(list(geom.exterior.coords))
#     # If the geometry is a MultiPolygon, get the vertices of its component Polygons
#     elif geom.geom_type == 'MultiPolygon':
#         st.write("MultiPolygon")
#         # for polygon in geom:
#         #     vertices.extend(list(polygon.exterior.coords))

# st.write(vertices)


# # Convert the MultiPolygon to a Polygon
# merged_geom = unary_union(gdf.geometry)

# # Get the vertices of each geometry in the GeoDataFrame
# vertices = list(merged_geom.exterior.coords)

# print(vertices)

# Create a map of the area
lidarArea = 'NY_NewYorkCity/'

st.markdown("## Downloading data from USGS for Central Park")

#Get Bounding Box of RI_gdf
lon_min, lat_min, lon_max, lat_max = Reduced_gdf.total_bounds

#Print the Bounding Box
st.write("Bounding Box in lat,long of Central Park:", lat_min, lon_min, lat_max, lon_max)

# Get center of Bounding Box
lat = (lat_min + lat_max) / 2
lon = (lon_min + lon_max) / 2

# Get the point cloud data
Raw_lidar_df, epsg_no = stackTiles_NotLimited(lat,lon,prefix=lidarArea)

#Convert Bounding Box to epsg_no
minX, minY = convertLatLon(lat_min, lon_min, epsg_no)
maxX, maxY = convertLatLon(lat_max, lon_max, epsg_no)

# Print the Bounding Box
st.write(f"Bounding Box in {epsg_no}:", minX, minY, maxX, maxY)
#Area of Bounding Box of RI_gdf
area = (maxX - minX) * (maxY - minY)
#Print the Area
st.write("Area of Bounding Box of Central Park:", area)

# Write the number of points found
st.write(f"Total Number of Raw Points {len(Raw_lidar_df)}")

#make a 3d plot of the points
#Raw_lidar_df = Raw_lidar_df[::10]

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

st.markdown("### Only Central Park Points")

# Get the points within the bounding box from the Raw_lidar_df
points_in_box = Raw_lidar_df[(Raw_lidar_df['X'] >= minX) & (Raw_lidar_df['X'] <= maxX) & (Raw_lidar_df['Y'] >= minY) & (Raw_lidar_df['Y'] <= maxY)]

# Write the number of points found
st.write(f"Number of Points in Bounding Box {len(points_in_box)}")

# Create 3D scatter plot using Plotly with classification-based colors

fig = px.scatter_3d(points_in_box, x='X', y='Y', z='Z', color='Z',
                    hover_data=['X', 'Y', 'Z', 'class'],
                    color_continuous_scale=px.colors.sequential.Viridis,
                    width=400, height=500)

fig.update_traces(marker=dict(size=1.2))
fig.update_layout(scene=dict(aspectmode='data'), showlegend=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

st.plotly_chart(fig, use_container_width=True)


# Create a KDTree from the points
#tree = cKDTree(Raw_lidar_df[['X', 'Y']].to_numpy())

# # Get the bounding box of the GeoDataFrame
# bbox = RI_gdf.total_bounds

# print(bbox)  # (minx, miny, maxx, maxy)

# # Get the bounding box of the polygon
# lon_min, lat_min, lon_max, lat_max = bbox

# st.write(f"Bounding Box in lat lon : {lon_min, lat_min, lon_max, lat_max}")

# minX, minY = convertLatLon(lat_min, lon_min, epsg_no)
# maxX, maxY = convertLatLon(lat_max, lon_max, epsg_no)

# st.write(f"Bounding Box in X Y : {minX, minY, maxX, maxY}")

# # Get the points within the bounding box from the Raw_lidar_df
# points_in_box_df = Raw_lidar_df[(Raw_lidar_df['X'] >= minX) & (Raw_lidar_df['X'] <= maxX) & (Raw_lidar_df['Y'] >= minY) & (Raw_lidar_df['Y'] <= maxY)]

# st.markdown("###Roosevelt Island Points")

# st.write(points_in_box_df)

# # Create 3D scatter plot using Plotly with classification-based colors
# fig = px.scatter_3d(points_in_box_df, x='X', y='Y', z='Z', color='Z',
#                     hover_data=['X', 'Y', 'Z', 'class'],
#                     color_continuous_scale=px.colors.sequential.Viridis,
#                     width=400, height=500)

# fig.update_traces(marker=dict(size=1.2))
# fig.update_layout(scene=dict(aspectmode='data'), showlegend=False)
# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showticklabels=False)


# st.plotly_chart(fig, use_container_width=True)


#Transform all the points in Raw_lidar_df to lat and lon using apply() and map()

# Raw_lidar_df[['Lat', 'Lon']] = Raw_lidar_df.apply(lambda row: pd.Series(convertXY(row['X'], row['Y'])), axis=1)

# #Create a geometry column from the lat and lon columns
# Raw_lidar_df['geometry'] = Raw_lidar_df.apply(lambda row: Point(row['Lon'], row['Lat']), axis=1)

# #Create a geodataframe from the lidar dataframe
# Raw_lidar_gdf = gpd.GeoDataFrame(Raw_lidar_df, geometry='geometry')

# check is Point(X,Y) is within the RI_gdf

# for index, row in Raw_lidar_df.iterrows():
#     lat, lon = convertXY(row['X'], row['Y'])
    
#     polygon = RI_gdf.iloc[0]['geometry']

#     #check if lat and lon are within the polygon
#     if polygon.contains(Point(lon, lat)):
#         print("Point is within the polygon")
#     else:
#         print("No")

# Create a new dataframe with only the points that are within the RI_gdf
#RI_lidar_df = gpd.sjoin(Raw_lidar_gdf, RI_gdf, op='within')

# Create 3D scatter plot using Plotly with classification-based colors
# fig = px.scatter_3d(RI_lidar_df, x='X', y='Y', z='Z', color='Z',
#                     hover_data=['X', 'Y', 'Z', 'class'],
#                     color_continuous_scale=px.colors.sequential.Viridis,
#                     width=400, height=500)

# fig.update_traces(marker=dict(size=1.2))
# fig.update_layout(scene=dict(aspectmode='data'), showlegend=False)
# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showticklabels=False)

# Create an R-tree index for the RI_gdf
# idx = index.Index()
# for i, geometry in enumerate(RI_gdf.geometry):
#     idx.insert(i, geometry.bounds)

# # Find the points in Raw_lidar_gdf that fall within the geometries of RI_gdf using the R-tree index
# hits = []
# for i, geometry in enumerate(Raw_lidar_gdf.geometry):
#     for j in idx.intersection(geometry.bounds):
#         if geometry.within(RI_gdf.iloc[j].geometry):
#             hits.append(i)

# # Create a new dataframe with only the points that fall within the geometries of RI_gdf
# boounded_RI_lidar_df = Raw_lidar_gdf.iloc[hits]

# # Create 3D scatter plot using Plotly with classification-based colors

# fig = px.scatter_3d(boounded_RI_lidar_df, x='X', y='Y', z='Z', color='Z',
#                     hover_data=['X', 'Y', 'Z', 'class'],
#                     color_continuous_scale=px.colors.sequential.Viridis,
#                     width=400, height=500)

# fig.update_traces(marker=dict(size=1.2))
# fig.update_layout(scene=dict(aspectmode='data'), showlegend=False)
# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showticklabels=False)

# st.plotly_chart(fig, use_container_width=True)

# st.plotly_chart(fig, use_container_width=True)


# st.write(RI_new_lidar_df)

#Perform a spatial join between the two dataframes
#joined = gpd.sjoin(gpd.GeoDataFrame(RI_lidar_df, geometry=gpd.points_from_xy(RI_lidar_df.X, RI_lidar_df.Y)), RI_gdf, op='within')

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
