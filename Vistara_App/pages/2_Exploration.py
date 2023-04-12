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



#st.write("Gorund Point data:", Gp_ldf)



# try:

#     stime = time.time()


#     Gpoints, NGpoints, Elevation = Extract_GroundData(f, year, lasfiles_folderpath=fpath)

#     #plotting ground poitns found
#     p1 = Gpoints
#     p2 = NGpoints
#     All_points_1 = np.concatenate((p1, p2), axis=0)
#     rgb_p1 =  [[1,0,0]]*len(p1) #Set red colour
#     rgb_p2 = [[255,255,255]]*len(p2) #set green colour - Classified tree points
#     All_rgb = np.concatenate((rgb_p1, rgb_p2,), axis=0)

#     v = pptk.viewer(All_points_1, All_rgb)
#     v.set(show_grid=False)
#     v.set(show_axis=False)
#     v.set(bg_color = [0,0,0,1])
#     v.set(point_size = 0.04)

#     TakeScreenShot(All_points_1,v,f,year,"Ground_Classification")

#     logging.info("MR Classification Algorithm initated for : "+f)

#     lidar_df, rawpoints, MR_df, SR_df = PreprocessLasFile(f, year, lasfiles_folderpath=fpath)

#     Approx_locations_str, T_lat, T_lon = Log_TileLocation(MR_df)
#     logging.info("Approximate Location of %s : %s", f,Approx_locations_str)

#     #lasTile class
#     TileObj_SR = MRC.MR_class(SR_df,TileDivision=10) #Single Return Points
#     TileObj_MR = MRC.MR_class(MR_df,TileDivision=10) #Multiple Return Points

#     #Serialized Creation of Lidar Subtiles
#     lidar_TilesubsetArr = TileObj_MR.Get_subtileArray()

#     All_eps = [] #Stores all eps values by tile id
#     N_Neighbours = 12
#     subT_ID = 0
#     TileDivision =10
#     EPS_distribution_df = pd.DataFrame(columns=['T_ID', 'T_lat', 'T_lon', 'subT_ID', 'subT_lat','subT_lon','EPS'])

#     for row in range(TileDivision):
#         for col in range(TileDivision):

#             if(len(lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()) > N_Neighbours):

#                 cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]
#                 subtile_location_str, subT_lat, subT_long = Log_TileLocation(cluster_df)
#                 subtile_eps = Get_eps_NN_KneeMethod(cluster_df)
#                 All_eps.append(subtile_eps)

#             EPS_dist_df_row = [f,T_lat,T_lon]
#             EPS_dist_df_row.append(subT_ID)
#             EPS_dist_df_row.append(subT_lat)
#             EPS_dist_df_row.append(subT_long)
#             EPS_dist_df_row.append(subtile_eps)

#             EPS_distribution_df.loc[len(EPS_distribution_df.index)] = EPS_dist_df_row
            
#             subT_ID = subT_ID + 1

#     Optimal_EPS = np.mean(All_eps)
#     logging.info("Avg EPS for %s : %s",f,Optimal_EPS)

#     EPS_CSV_filename = 'Spatial_HP_Distribution_'+f+"_"+str(year)+'.csv'
#     EPS_CSV_dir = "Datasets/"+"Package_Generated/"+f[:-4]+"/"+str(year)+"/LiDAR_HP_MATRIX_"+f[:-4]+"/"
#     # Check whether the specified EPS_CSV_dir exists or not
#     isExist = os.path.exists(EPS_CSV_dir)

#     if not isExist:
#     # Create a new directory because it does not exist 
#         os.makedirs(EPS_CSV_dir)

#     logging.info("MR - T_ID : %s - ACTION: HP_MATRIX CSV file Created",f)
#     EPS_distribution_df.to_csv(EPS_CSV_dir+EPS_CSV_filename)

#     Tilecounter = 0
#     Trees_Buffer = []
#     N_Neighbours = 12

#     for row in range(TileDivision):
#         for col in range(TileDivision):

#             #print('-'*40)
            
#             #print("TILE ID : ",Tilecounter)
#             Tilecounter = Tilecounter + 1

#             if (len(lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()) > N_Neighbours):

#                 cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]
#                 tile_eps = Get_eps_NN_KneeMethod(cluster_df) #round(Optimal_EPS,2)
#                 #print(tile_eps)
#                 tile_segment_points = lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()
#                 subTileTree_Points,  _ = TileObj_MR.Classify_MultipleReturns(tile_segment_points,tile_eps)

#                 for t in subTileTree_Points:
#                     Trees_Buffer.append(t)
                
#                 logging.info("MR - T_ID : %s - ACTION: Trees Added to - S_ID : %d",f,Tilecounter)
            
#             else:
#                 logging.warn("Empty Tileset Found")

#     Trees_Buffer = np.array(Trees_Buffer)

#     db = DBSCAN(eps=np.mean(EPS_distribution_df.EPS), min_samples=30).fit(Trees_Buffer)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     DB_labels = db.labels_

#     rgb_array = assign_colors(Trees_Buffer, DB_labels)
#     v = pptk.viewer(Trees_Buffer)
#     v.attributes(rgb_array/255)
#     v.set(show_grid=False)
#     v.set(show_axis=False)
#     v.set(bg_color = [0,0,0,1])
#     v.set(point_size = 0.04)

#     TakeScreenShot(Trees_Buffer,v,f,year,"MR_Points")

#     MR_TreesDf = pd.DataFrame(Trees_Buffer, columns=["X","Y","Z"])
#     MR_TreesDf["Cluster_Labels"] = DB_labels

#     cluster_sizes = np.bincount(DB_labels[DB_labels != -1]) #exclude outliers
#     C_mean = np.mean(cluster_sizes)
#     C_std = np.std(cluster_sizes)

#     logging.info("MR - T_ID : %s - Stats : ClusterSize_mean: %s , ClusterSize_std : %s",f,C_mean,C_std)

#     #Get Largest clusters
#     #Parks clusters -> clusters with > 2 std from mean points
#     Park_Clusters = np.where(cluster_sizes > C_mean + 2*C_std)[0]
#     Filtered_cluster_sizes = np.where(cluster_sizes < C_mean + 2*C_std)[0]
#     FC_mean = np.mean(Filtered_cluster_sizes)
#     FC_std = np.std(Filtered_cluster_sizes)
    
#     logging.info("MR - T_ID : %s - STATS [PARK REMOVED]: ClusterSize_mean: %s , ClusterSize_std : %s",f,FC_mean,FC_std)

#     #Recalculating thresholds as park cluster sizes make the std >> mean 
#     # and hence no clusters were showing up when mean - 1*std was taken

#     #small clusters -> clusters with <  1 std from mean points
#     Small_Clusters = np.where(cluster_sizes < FC_mean + -1*FC_std)[0]
#     #large clusters -> [0.05 std , 2 std]
#     Large_Clusters = np.where((cluster_sizes > FC_mean + 0.5*FC_std)
#                             &
#                             (cluster_sizes < FC_mean + 1.5*FC_std))[0]
#     #Acceptable Clusters -> [-1 std to + 0.05 std]  
#     Accepted_clusters = np.where((cluster_sizes > FC_mean + -1*FC_std)
#                             &
#                             (cluster_sizes < FC_mean + 0.5*FC_std))[0] 
    
#     #Processing Park Clusters
#     Park_CLusterPoints = Get_ClusterPoints(MR_TreesDf, Park_Clusters)
#     Small_ClustersPoints = Get_ClusterPoints(MR_TreesDf, Small_Clusters)
#     Large_ClustersPoints = Get_ClusterPoints(MR_TreesDf, Large_Clusters)
#     Accepted_clustersPoints = Get_ClusterPoints(MR_TreesDf, Accepted_clusters)

#     p1 = Accepted_clustersPoints
#     p2 = Large_ClustersPoints #Park_CLusterPoints #Accepted_clustersPoints
#     All_points_1 = np.concatenate((p1, p2), axis=0)
#     rgb_p1 = [[1,0,0]]*len(p1) #set red colour
#     rgb_p2 =  [[255,255,255]]*len(p2) #Set white colour 
#     All_rgb = np.concatenate((rgb_p1, rgb_p2), axis=0)

#     v = pptk.viewer(All_points_1, All_rgb)
#     v.set(show_grid=False)
#     v.set(show_axis=False)
#     v.set(bg_color = [0,0,0,1])
#     v.set(point_size = 0.04)


#     TakeScreenShot(All_points_1,v,f,year,"ClusterSegmentation_(WL_RA)")

#     #Get Clusters from Local Maximas
#     PC_labels = []
#     peaks, _ = find_peaks(Park_CLusterPoints[:,2], distance=500,prominence=3)
#     # Get the x, y, and z coordinates of the local maximas
#     local_maximas = Park_CLusterPoints[peaks]

#     #Perform NN
#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(local_maximas)
#     distances, indices = nbrs.kneighbors(Park_CLusterPoints)

#     PC_labels = pd.Series(indices.flatten())

#     rgb_array = assign_colors(Park_CLusterPoints, PC_labels)
#     v = pptk.viewer(Park_CLusterPoints)
#     v.attributes(rgb_array/255)
#     v.set(show_grid=False)
#     v.set(show_axis=False)
#     v.set(bg_color = [0,0,0,1])
#     v.set(point_size = 0.04)

#     TakeScreenShot(Park_CLusterPoints,v,f,year,"Park_Clusters")

#     #Processing Regular Clusters
#     db = DBSCAN(eps=1.5, min_samples=30).fit(Accepted_clustersPoints)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     AC_labels = db.labels_

#     rgb_array = assign_colors(Accepted_clustersPoints, AC_labels)
#     v = pptk.viewer(Accepted_clustersPoints)
#     v.attributes(rgb_array/255)
#     v.set(show_grid=False)
#     v.set(show_axis=False)
#     v.set(bg_color = [0,0,0,1])
#     v.set(point_size = 0.04)

#     TakeScreenShot(Accepted_clustersPoints,v,f,year,"Regular_TreeClusters")

#     #Processing Large Clusters

#     LC_df = pd.DataFrame(Large_ClustersPoints,columns=["X","Y","Z"])

#     maxSearch = LC_df
#     maxSearch['seeds'] = 1
#     xs = maxSearch['X'].to_numpy()
#     ys = maxSearch['Y'].to_numpy()
#     zs = maxSearch['Z'].to_numpy()
#     seeds = maxSearch['seeds'].to_numpy()

#     neighborhood = 4
#     changes = 1
#     while changes > 0:
#         changes = 0
#         i = 0
#         for x,y,z,seed in zip(xs,ys,zs,seeds):
#             if seed == 1:
#                 zsearch = zs[xs > x - neighborhood]
#                 xsearch = xs[xs > x - neighborhood]
#                 ysearch = ys[xs > x - neighborhood]
                
#                 zsearch = zsearch[xsearch < x + neighborhood]
#                 ysearch = ysearch[xsearch < x + neighborhood]
                
#                 zsearch = zsearch[ysearch > y - neighborhood]
#                 ysearch = ysearch[ysearch > y - neighborhood]
                
#                 zsearch = zsearch[ysearch < y + neighborhood]

#                 zmax = np.max(zsearch)
#                 if z < zmax:
#                     seeds[i] = 0
#                     changes += 1
#                 else:
#                     pass
#             else:
#                 pass
#             i+=1

#     maxSearch['seeds'] = seeds      
#     localMaxima = maxSearch[maxSearch['seeds']>0]

#     centers=localMaxima[['X','Y','Z']].to_numpy()
#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centers)
#     points = maxSearch[['X','Y','Z']].to_numpy()
#     distances, indices = nbrs.kneighbors(points)
#     maxSearch['nearestNeighbor'] = indices

#     LC_labels = pd.Series(indices.flatten())
#     rgb_array = assign_colors(Large_ClustersPoints, LC_labels)
#     v = pptk.viewer(Large_ClustersPoints)
#     v.attributes(rgb_array/255)
#     v.set(show_grid=False)
#     v.set(show_axis=False)
#     v.set(bg_color = [0,0,0,1])
#     v.set(point_size = 0.04)

#     TakeScreenShot(Accepted_clustersPoints,v,f,year,"Large_TreeClusters_NN")

#     logging.info("MR - T_ID : %s - STATS: Park Trees: %d ",f,len(np.unique(PC_labels)))
#     logging.info("MR - T_ID : %s - STATS: Large CLuster Trees: %d ",f,len(np.unique(LC_labels)))
#     logging.info("MR - T_ID : %s - STATS: Acceptable Trees: %d ",f,len(np.unique(AC_labels)))

#     Total_TreeCount = len(np.unique(PC_labels))+len(np.unique(LC_labels))+len(np.unique(AC_labels))
#     logging.info("MR - T_ID : %s - STATS: TOTAL Trees: %d ",f,Total_TreeCount)

#     NGpoints_df = pd.DataFrame(NGpoints,columns=["X","Y","Z"])

#     JSONfoldernName = "JSON_TreeData_"+f[:-4]

#     jfolderpath = "Datasets/" + "Package_Generated/"+f[:-4]+"/"+str(year)+"/" + JSONfoldernName +"/"

#     # Check whether the specified jfolderpath exists or not
#     isExist = os.path.exists(jfolderpath)

#     if not isExist:
#     # Create a new directory because it does not exist 
#         os.makedirs(jfolderpath)

#     # labels = [PC_labels,LC_labels,AC_labels]
#     # point_arr = [Park_CLusterPoints, Large_ClustersPoints, Accepted_clustersPoints]
#     # arr = Park_CLusterPoints
#     Global_TreeCounter = 0


#     las_fileID = f[:-4]

#     Extracted_SRpoints = []
#     Extracted_NTpoints = []


#     Global_TreeCounter = Store_TreeCluster_toJSON(f, year,
#                                 Park_CLusterPoints, PC_labels,
#                                 NGpoints_df, 
#                                 Extracted_SRpoints, Extracted_NTpoints,
#                                 Global_TreeCounter,
#                                 Elevation,
#                                 jfolderpath,isPark=True)

#     Global_TreeCounter = Store_TreeCluster_toJSON(f, year,
#                                 Large_ClustersPoints, LC_labels,
#                                 NGpoints_df, 
#                                 Extracted_SRpoints, Extracted_NTpoints,
#                                 Global_TreeCounter + 1,
#                                 Elevation,
#                                 jfolderpath)

#     Global_TreeCounter = Store_TreeCluster_toJSON(f, year,
#                                 Accepted_clustersPoints, AC_labels,
#                                 NGpoints_df, 
#                                 Extracted_SRpoints, Extracted_NTpoints,
#                                 Global_TreeCounter + 1,
#                                 Elevation,
#                                 jfolderpath)

#     #Remove NT points from ground points
#     #Extracted_NTpoints = np.array([i for i in Extracted_NTpoints if i not in Gpoints])

#     All_Classified_points = np.concatenate((Gpoints, Trees_Buffer,Extracted_SRpoints,Extracted_NTpoints), axis=0)

#     #Classification codes were defined by the American Society for Photogrammetry and Remote Sensing (ASPRS)
#     #  for LAS formats 1.1, 1.2, 1.3, and 1.4

#     Classifiedpoints_df = pd.DataFrame(All_Classified_points,columns=["X","Y","Z"])
#     Classifiedpoints_df = Classifiedpoints_df.drop_duplicates()
#     Original_lidarpoints_df = lidar_df.iloc[:,:3]

#     df1=Original_lidarpoints_df
#     df2=Classifiedpoints_df

#     df_merged = df1.merge(df2, how="left", on=['X','Y','Z'], indicator=True)
#     Unclassified_Points_df =  df_merged.query("_merge == 'left_only'")[['X','Y','Z']]

#     Unclassified_Points = Unclassified_Points_df.to_numpy()

#     class_1Labels = [2]*(len(Gpoints))
#     class_2Labels = [5]*(len(Trees_Buffer))
#     class_3Labels = [4]*(len(Extracted_SRpoints))
#     class_4Labels = [6]*(len(Extracted_NTpoints))
#     class_5Labels = [1]*(len(Unclassified_Points))

#     final_labels = np.concatenate((class_1Labels,class_2Labels,class_3Labels,class_4Labels,class_5Labels),axis=0)
#     All_points = np.concatenate((Gpoints, Trees_Buffer,Extracted_SRpoints,Extracted_NTpoints,Unclassified_Points), axis=0)

#     clippedLasNumpy = All_points
#     las = laspy.create(file_version="1.4", point_format=3)

#     Xscale = 0.01
#     Yscale = 0.01
#     Zscale = 0.01
#     Xoffset = 0
#     Yoffset = 0
#     Zoffset = 0

#     las.header.offsets = [Xoffset,Yoffset,Zoffset]
#     las.header.scales = [Xscale,Yscale,Zscale]
#     las.x = clippedLasNumpy[:, 0]
#     las.y = clippedLasNumpy[:, 1]
#     las.z = clippedLasNumpy[:, 2]
#     las.intensity = [0]*len(clippedLasNumpy)
#     las.classification =  final_labels
#     las.return_number =  [0]*len(clippedLasNumpy)
#     las.number_of_returns =  [0]*len(clippedLasNumpy)

#     generated_lasfolderpath =  "Datasets/" + "Package_Generated/"+f[:-4]+"/"+str(year)+"/LasClassified_"+f[:-4]+"/"
#     # Check whether the specified generated_lasfolderpath exists or not
#     isExist = os.path.exists(generated_lasfolderpath)
#     if not isExist:
#     # Create a new directory because it does not exist 
#         os.makedirs(generated_lasfolderpath)
        
#     generated_lasfilename = "lasFile_Reconstructed_"+f

#     las.write(generated_lasfolderpath+generated_lasfilename)

#     #Plotting new classified las file

#     Gen_las = laspy.read(generated_lasfolderpath+generated_lasfilename)

#     Xscale = Gen_las.header.x_scale
#     Yscale = Gen_las.header.y_scale
#     Zscale = Gen_las.header.z_scale

#     Xoffset = Gen_las.header.x_offset
#     Yoffset = Gen_las.header.y_offset
#     Zoffset = Gen_las.header.z_offset

#     Gen_lidarpoints = np.array(
#         ( (Gen_las.X*1.00) + Xoffset,  # convert ft to m and correct measurement
#         (Gen_las.Y*1.00) + Yoffset,
#         (Gen_las.Z*1.00) + Zoffset,
#         Gen_las.intensity,
#         Gen_las.classification,
#         Gen_las.return_number, 
#         Gen_las.number_of_returns)).transpose()
#     G_lidar_df = pd.DataFrame(Gen_lidarpoints , columns=['X','Y','Z','intensity','classification','return_number','number_of_returns'])


#     G_las_Gpoints = G_lidar_df.iloc[:,:3][G_lidar_df["classification"] == 2].to_numpy()
#     G_las_Tpoints = G_lidar_df.iloc[:,:3][G_lidar_df["classification"] == 5].to_numpy()
#     G_las_SRpoints = G_lidar_df.iloc[:,:3][G_lidar_df["classification"] == 4].to_numpy()
#     G_las_NTpoints = G_lidar_df.iloc[:,:3][G_lidar_df["classification"] == 6].to_numpy()
#     G_las_NCpoints = G_lidar_df.iloc[:,:3][G_lidar_df["classification"] == 1].to_numpy()

#     #plotting inlier and outlier
#     All_points_1 = np.concatenate((G_las_Gpoints, G_las_Tpoints,G_las_SRpoints,G_las_NTpoints,G_las_NCpoints), axis=0)
#     rgb_Ground =  [[1,0,0]]*len(G_las_Gpoints) #Set red colour
#     rgb_Tree = [[0,1,0]]*len(G_las_Tpoints) #set green colour
#     rgb_SR = [[0,0,1]]*len(G_las_SRpoints) #set blue colour
#     rgb_NT = [[255,255,255]]*len(G_las_NTpoints) #set white colour
#     rgb_NC = [[255,255,0]]*len(G_las_NCpoints) #set cyan colour
#     All_rgb = np.concatenate((rgb_Ground, rgb_Tree,rgb_SR,rgb_NT,rgb_NC), axis=0)

#     #Red - Inlier - ground plane , Green - Outlier
#     v = pptk.viewer(All_points_1, All_rgb)
#     v.set(show_grid=False)
#     v.set(show_axis=False)
#     v.set(bg_color = [0,0,0,1])
#     v.set(point_size = 0.04)

#     TakeScreenShot(All_points_1,v,f,year,"Classified_LasFile")

#     etime = time.time()
#     logging.info("Completed file: %s - Total Time Taken : %d", f, etime - stime)

# except Exception as e:
#     logging.error("ERROR : %s",str(e))