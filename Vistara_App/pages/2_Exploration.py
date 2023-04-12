import streamlit as st
import plotly.express as px


# User Defined Package

import src.modules.utils as util

import src.modules.utils as util
import src.modules.MultipleReturnsClassification as MRC
import src.modules.LasFilePreprocessing as LFP
import src.modules.ExtractGroundPlane as GP


from multiprocessing import Pool
import logging
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score # How best can we seperate clusters
import time
from scipy import spatial
import re #parsing log files
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import Point
import json
import laspy


st.markdown("## Ground Plane Extraction")

def InitiateLogger(filename="Vistara")-> None:

    LoggerPath = "LogFile/"

    print("Logger Folder Path : ",LoggerPath)
    # Check whether the specified pptk_capture_path exists or not
    isExist = os.path.exists(LoggerPath)

    if not isExist:
    # Create a new directory because it does not exist 
        os.makedirs(LoggerPath)

    logfilename = LoggerPath + 'Exploration_'+filename+'.log' 
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=logfilename, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

#MR Classification
def Log_TileLocation(MR_df):

    #Print Lat , Long
    lat, lon = np.mean(MR_df.X.to_numpy()), np.mean(MR_df.Y.to_numpy()) 

    location_str = str(lat)+","+str(lon)

    return location_str, lat,lon

def Get_eps_NN_KneeMethod(cluster_df, N_neighbors = 12, display_plot=False):

    nearest_neighbors = NearestNeighbors(n_neighbors=N_neighbors)
    neighbors = nearest_neighbors.fit(cluster_df)
    distances, indices = neighbors.kneighbors(cluster_df)
    distances = np.sort(distances[:,N_neighbors-1], axis=0)

    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    if (display_plot):
        fig = plt.figure(figsize=(5, 5))
        knee.plot_knee()
        plt.xlabel("Points")
        plt.ylabel("Distance")
        print(distances[knee.knee])
    
    return distances[knee.knee]


#########################################################################


#Load in Extracted Lidar Data
lidar_df = st.session_state["Extracted_Lidar_Data"]
filename = st.session_state["filename"]

TileDivision = 1

#Initate logger
InitiateLogger()
logging.info("TerraVide lidar processing Initated")

#MR_df = LFP.Get_MRpoints(lidar_df) Not needed for ground points
SR_df = LFP.Get_SRpoints(lidar_df)

#lasTile class
TileObj = LFP.lasTile(SR_df,TileDivision)

#Serialized
s_start = time.time()

lidar_TilesubsetArr = TileObj.Get_subtileArray()

s_end = time.time()
stime = s_end - s_start
logging.info("Extraction of Subtile Matrix Buffer Serial Time for = %d",stime)

#Ground Plane Classifcation - Serial Implementation
g_start = time.time()
Potential_Ground_Points = []
Other_points = []

GP_obj = GP.GP_class()

for row in range(TileDivision):
    for col in range(TileDivision):

        tile_segment_points = lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()

        local_Ground_Points, Nlocal_Ground_Points = GP_obj.Extract_GroundPoints(tile_segment_points)

        for k in local_Ground_Points:
            Potential_Ground_Points.append(k) #append points which may be potentially ground points
        for l in Nlocal_Ground_Points:
            Other_points.append(l) #append points which may be potentially ground points

if len(Potential_Ground_Points) == 0:
    logging.info("No Ground Points Found")
    st.write("No Ground Points Found")

    #Initiate empty 3d arrays
    Potential_Ground_Points = np.array([None,None,None])
    #st.stop()

Potential_Ground_Points = np.array(Potential_Ground_Points)
Other_points = np.array(Other_points)

# st.write(Potential_Ground_Points.shape)
# st.write(Other_points.shape)

g_end = time.time()
gtime = g_end - g_start
logging.info("Ground Point Extraction Algorithm Serial Time for %s = %d",filename,gtime)

#Create a dataframe for Potential_Ground_Points and Other_points for plotting

GP_df = pd.DataFrame(Potential_Ground_Points, columns=['X', 'Y', 'Z'])
GP_df['class'] = 'Ground'
NG_df = pd.DataFrame(Other_points, columns=['X', 'Y', 'Z'])
NG_df['class'] = 'Non-Ground'

Gp_ldf = pd.concat([GP_df, NG_df], ignore_index=True)

#Plotting Ground Points

# Create 3D scatter plot using Plotly with classification-based colors
fig = px.scatter_3d(Gp_ldf, x='X', y='Y', z='Z', color='class',
                    color_discrete_map={'Ground': 'blue', 'Non-Ground': 'red'},
                    hover_data=['X', 'Y', 'Z', 'class'])

fig.update_traces(marker=dict(size=1.2))
fig.update_layout(scene=dict(aspectmode='data'))

st.plotly_chart(fig)

#Ask user if they want to see the dataframes

if st.checkbox("Show Ground Point Dataframe"):
    st.write("Gorund Point data:", Gp_ldf)


st.markdown("## Clustering")

logging.info("MR Classification Algorithm initated for : "+filename)

#Get rows from lidar_df which are not in GP_df using X,Y,Z
R_lidar_df = lidar_df[~lidar_df[['X', 'Y', 'Z']].isin(GP_df[['X', 'Y', 'Z']]).all(axis=1)]

#Extract MR and SR points from Dataframe
LasHandling = MRC.LFP
MR_df = LasHandling.Get_MRpoints(R_lidar_df)
SR_df = LasHandling.Get_SRpoints(R_lidar_df)

#lidar_df, rawpoints, MR_df, SR_df = PreprocessLasFile(f, year, lasfiles_folderpath=fpath)

#lasTile class
TileObj_SR = MRC.MR_class(SR_df,TileDivision) #Single Return Points
TileObj_MR = MRC.MR_class(MR_df,TileDivision) #Multiple Return Points

#Serialized Creation of Lidar Subtiles
lidar_TilesubsetArr = TileObj_MR.Get_subtileArray()

st.write(lidar_TilesubsetArr[0][0])

All_eps = [] #Stores all eps values by tile id
N_Neighbours = 12
subT_ID = 0
TileDivision =10
EPS_distribution_df = pd.DataFrame(columns=['T_ID', 'T_lat', 'T_lon', 'subT_ID', 'subT_lat','subT_lon','EPS'])


if(len(lidar_TilesubsetArr[0][0].iloc[:,:3].to_numpy()) > N_Neighbours):

    cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]
    subtile_location_str, subT_lat, subT_long = Log_TileLocation(cluster_df)
    subtile_eps = Get_eps_NN_KneeMethod(cluster_df)
    All_eps.append(subtile_eps)

EPS_dist_df_row = [filename,subT_lat,subT_long]
EPS_dist_df_row.append(subT_ID)
EPS_dist_df_row.append(subT_lat)
EPS_dist_df_row.append(subT_lat)
EPS_dist_df_row.append(subtile_eps)

EPS_distribution_df.loc[len(EPS_distribution_df.index)] = EPS_dist_df_row

subT_ID = subT_ID + 1

Optimal_EPS = np.mean(All_eps)
logging.info("Avg EPS for %s : %s",filename,Optimal_EPS)

EPS_CSV_filename = 'Spatial_HP_Distribution_'+filename+'.csv'
EPS_CSV_dir = "Datasets/"+"Package_Generated/"+filename+"/LiDAR_HP_MATRIX_"+filename+"/"
# Check whether the specified EPS_CSV_dir exists or not
isExist = os.path.exists(EPS_CSV_dir)

if not isExist:
# Create a new directory because it does not exist 
    os.makedirs(EPS_CSV_dir)

logging.info("MR - T_ID : %s - ACTION: HP_MATRIX CSV file Created",filename)
EPS_distribution_df.to_csv(EPS_CSV_dir+EPS_CSV_filename)

Tilecounter = 0
Trees_Buffer = []
N_Neighbours = 12

# for row in range(TileDivision):
#     for col in range(TileDivision):

#         #print('-'*40)
        
#         #print("TILE ID : ",Tilecounter)
#         Tilecounter = Tilecounter + 1

row = 0
col = 0

if (len(lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()) > N_Neighbours):

    cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]
    tile_eps = Get_eps_NN_KneeMethod(cluster_df) #round(Optimal_EPS,2)
    st.write(tile_eps)
    #print(tile_eps)
    tile_segment_points = lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()
    subTileTree_Points,  _ = TileObj_MR.Classify_MultipleReturns(tile_segment_points,tile_eps)

    for t in subTileTree_Points:
        Trees_Buffer.append(t)
    
    logging.info("MR - T_ID : %s - ACTION: Trees Added to - S_ID : %d",filename,Tilecounter)

else:
    logging.warn("Empty Tileset Found")

Trees_Buffer = np.array(Trees_Buffer)

db = DBSCAN(eps=np.mean(EPS_distribution_df.EPS), min_samples=30).fit(Trees_Buffer)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
DB_labels = db.labels_

MR_TreesDf = pd.DataFrame(Trees_Buffer, columns=["X","Y","Z"])
MR_TreesDf["class"] = DB_labels

#Total Trees = number of unique classes
Total_Trees = len(MR_TreesDf["class"].unique())
logging.info("MR - T_ID : %s - ACTION: Total Trees : %d",filename,Total_Trees)

#write it to the app in bold
st.write("Total Trees : ",Total_Trees)


#Plotting the clustered trees

fig = px.scatter_3d(MR_TreesDf, x='X', y='Y', z='Z',
                color='class', opacity=0.5)
fig.update_traces(marker=dict(size=1.2))
fig.update_layout(scene=dict(aspectmode='data'))

st.plotly_chart(fig)

#Ask user if they want to see the dataframes

if st.checkbox("Show Clustered Trees Dataframe"):
    st.write("Clustered Trees data:", MR_TreesDf)

#Comine Ground and Clustered Trees Dataframes
#modify the class column of the GP_df dataframe
GP_df["class"] = -1
Combined_df = pd.concat([GP_df, MR_TreesDf], ignore_index=True)

#Generate a color map of default colors


# st.write(color_map)
#Convert classification types to string
Combined_df["class"] = Combined_df["class"].astype(str)

# color_map = {
#     '0': 'white',
#     '1': 'red',
#     '2': 'yellow',
#     '3': 'orange',
#     '4': 'green'
# }

color_map = {}
for i in range(len(Combined_df["class"].unique())):
    color_map[str(i)] = px.colors.qualitative.Plotly[i]

#Plot the combined dataframe

fig = px.scatter_3d(Combined_df, x='X', y='Y', z='Z',
                color='class', color_discrete_map=color_map)
fig.update_traces(marker=dict(size=1.2))
fig.update_layout(scene=dict(aspectmode='data'))

st.plotly_chart(fig)


#Write the combined dataframe if user wants to see it
if st.checkbox("Show Combined Dataframe"):
    st.write("Combined data:", Combined_df)
