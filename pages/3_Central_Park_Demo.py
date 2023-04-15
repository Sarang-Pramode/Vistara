import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
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


import src.modules.utils as util
import src.modules.MultipleReturnsClassification as MRC
import src.modules.LasFilePreprocessing as LFP
import src.modules.ExtractGroundPlane as GP


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
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
import alphashape


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

def stackTiles_NotLimited(lat,lon, boxSize=5000, prefix ='NY_NewYorkCity/'):  # 'NY_FingerLakes_1_2020/' #
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
            
    for depth in range(1,12):
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
    lidar_df = lidar_df[lidar_df['X'] <= x + boxSize ]
    lidar_df = lidar_df[lidar_df['X'] >= x - boxSize ]
    lidar_df = lidar_df[lidar_df['Y'] <= y + boxSize ]
    lidar_df = lidar_df[lidar_df['Y'] >= y - boxSize ]
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

###############################################################################

st.set_page_config(
        page_title="Central Park",
)

st.markdown("<h1 style='text-align: center; color: white;'>Central Park</h1>", unsafe_allow_html=True)


filepath = '2010 Census Tracts/geo_export_139fc905-b132-4c03-84d5-ae9e70dded42.shp'
gdf = gpd.read_file(filepath)

#rows_to_keep = [1625, 1980]
Reduced_gdf = gdf.loc[[2150]] # central Park

#Get Bounding Box of Reduced_gdf
lon_min, lat_min, lon_max, lat_max = Reduced_gdf.total_bounds

# Get the center of the bounding box
lat_c = (lat_min + lat_max) / 2
lon_c = (lon_min + lon_max) / 2

# Print the latitude and longitude
print("Latitude:", lat_c)
print("Longitude:", lon_c)

#Get Bounding Box of Central Park
lon_min, lat_min, lon_max, lat_max = Reduced_gdf.total_bounds

#Print the Bounding Box
#st.write("Bounding Box in lat,long of Central Park using total_bounds:", lat_min, lon_min, lat_max, lon_max)


#Plot RI_gdf
fig = px.choropleth_mapbox(
        Reduced_gdf,
        geojson=Reduced_gdf.geometry,
        locations=Reduced_gdf.index,
        mapbox_style='carto-positron',
        center={'lat': lat_c, 'lon':lon_c},
        zoom=12)


fig.update_layout(height=400, width=600, showlegend=False,
                margin=dict(l=0,r=0,b=0,t=0),
                paper_bgcolor="Black"
)

st.plotly_chart(fig)
###############################################################################

#Coordinates of Central Park (taken manually)

A = [40.76439182920556, -73.97303573603314]
B = [40.796890517282634, -73.94929586106859]
C = [40.800611517619174, -73.95819192353994]
D = [40.76821921884363, -73.98174547550886]

# Calculate the bounding box
min_lat = min(A[0], B[0], C[0], D[0])
max_lat = max(A[0], B[0], C[0], D[0])
min_lon = min(A[1], B[1], C[1], D[1])
max_lon = max(A[1], B[1], C[1], D[1])

# Print the bounding box coordinates
print("Bounding box coordinates of Central Park:")
print("Min Latitude:", min_lat)
print("Max Latitude:", max_lat)
print("Min Longitude:", min_lon)
print("Max Longitude:", max_lon)

#Take input from user on the resolution of data to be downloaded
BoxSize_input = 1800 #st.slider('Select the resolution of data to be downloaded', 0, 3000, 100)

#Add a button which ask the user to start the process
if st.button('Start TerraVide Classification'):

    st.markdown("## Downloading data from USGS for Central Park")

    # Create a map of the area
    lidarArea = 'NY_NewYorkCity/'

    # Get the point cloud data
    Raw_lidar_df, epsg_no = stackTiles_NotLimited(lat_c,lon_c,BoxSize_input,prefix=lidarArea)

    #Convert Bounding Box based on epsg_no
    minX, minY = convertLatLon(min_lat, min_lon, epsg_no)
    maxX, maxY = convertLatLon(max_lat, max_lon, epsg_no)

    # Write the number of points found
    st.write(f"Total Number of Raw Points {len(Raw_lidar_df)}")

    # Plot the raw point cloud
    # Create 3D scatter plot using Plotly with classification-based colors
    fig = px.scatter_3d(Raw_lidar_df[::2], x='X', y='Y', z='Z', color='Z',
                        hover_data=['X', 'Y', 'Z', 'class'],
                        width=400, height=500)

    fig.update_traces(marker=dict(size=1.2))
    fig.update_layout(scene=dict(aspectmode='data'), showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)


    st.plotly_chart(fig, use_container_width=True)


    #Load in Extracted Lidar Data
    lidar_df = Raw_lidar_df
    filename = st.session_state["filename"]

    TileDivision_g = 30

    #Initate logger
    InitiateLogger()
    logging.info("TerraVide lidar processing Initated")

    #MR_df = LFP.Get_MRpoints(lidar_df) Not needed for ground points
    SR_df = LFP.Get_SRpoints(lidar_df)

    #lasTile class
    TileObj = LFP.lasTile(SR_df,TileDivision=TileDivision_g)

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

    for row in range(TileDivision_g):
        for col in range(TileDivision_g):

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

    if Potential_Ground_Points.shape[0] == 0 or Other_points.shape[0] == 0:

        #Tell the user that no ground points were not able to be extracted on the streamlit page
        st.write("No Ground Points Found")

        #Direct the User back to Extraction Page
        st.write("Please go back to the Extraction Page and choose a different area of Interest")

    else:

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


        st.markdown("## Tree Canopy Classification Algorithm")

        logging.info("MR Classification Algorithm initated for : "+filename)

        #Get rows from lidar_df which are not in GP_df using X,Y,Z
        R_lidar_df = lidar_df[~lidar_df[['X', 'Y', 'Z']].isin(GP_df[['X', 'Y', 'Z']]).all(axis=1)]

        #Extract MR and SR points from Dataframe
        LasHandling = MRC.LFP
        MR_df = LasHandling.Get_MRpoints(R_lidar_df)
        SR_df = LasHandling.Get_SRpoints(R_lidar_df)

        #lidar_df, rawpoints, MR_df, SR_df = PreprocessLasFile(f, year, lasfiles_folderpath=fpath)

        TileDivision_t = 60

        #lasTile class
        TileObj_SR = MRC.MR_class(SR_df,TileDivision_t) #Single Return Points
        TileObj_MR = MRC.MR_class(MR_df,TileDivision_t) #Multiple Return Points

        #Serialized Creation of Lidar Subtiles
        lidar_TilesubsetArr = TileObj_MR.Get_subtileArray()

        #st.write(lidar_TilesubsetArr[0][0])

        # All_eps = [] #Stores all eps values by tile id
        # N_Neighbours = 12
        # subT_ID = 0
        # TileDivision =10
        # EPS_distribution_df = pd.DataFrame(columns=['T_ID', 'T_lat', 'T_lon', 'subT_ID', 'subT_lat','subT_lon','EPS'])


        # if(len(lidar_TilesubsetArr[0][0].iloc[:,:3].to_numpy()) > N_Neighbours):

        #     cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]
        #     subtile_location_str, subT_lat, subT_long = Log_TileLocation(cluster_df)
        #     subtile_eps = Get_eps_NN_KneeMethod(cluster_df)
        #     All_eps.append(subtile_eps)

        # EPS_dist_df_row = [filename,subT_lat,subT_long]
        # EPS_dist_df_row.append(subT_ID)
        # EPS_dist_df_row.append(subT_lat)
        # EPS_dist_df_row.append(subT_lat)
        # EPS_dist_df_row.append(subtile_eps)

        # EPS_distribution_df.loc[len(EPS_distribution_df.index)] = EPS_dist_df_row

        # subT_ID = subT_ID + 1

        # Optimal_EPS = np.mean(All_eps)
        # logging.info("Avg EPS for %s : %s",filename,Optimal_EPS)

        # EPS_CSV_filename = 'Spatial_HP_Distribution_'+filename+'.csv'
        # EPS_CSV_dir = "Datasets/"+"Package_Generated/"+filename+"/LiDAR_HP_MATRIX_"+filename+"/"
        # # Check whether the specified EPS_CSV_dir exists or not
        # isExist = os.path.exists(EPS_CSV_dir)

        # if not isExist:
        # # Create a new directory because it does not exist 
        #     os.makedirs(EPS_CSV_dir)

        # logging.info("MR - T_ID : %s - ACTION: HP_MATRIX CSV file Created",filename)
        # EPS_distribution_df.to_csv(EPS_CSV_dir+EPS_CSV_filename)

        Tilecounter = 0
        Trees_Buffer = []
        N_Neighbours = 12
        DB_labels = []

        for row in range(TileDivision_t):
            for col in range(TileDivision_t):

                #print('-'*40)
                
                #print("TILE ID : ",Tilecounter)
                Tilecounter = Tilecounter + 1

                if (len(lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()) > N_Neighbours):

                    cluster_df = lidar_TilesubsetArr[row][col].iloc[:,:3]
                    tile_eps = 30 #Get_eps_NN_KneeMethod(cluster_df) #round(Optimal_EPS,2)
                    #st.write(tile_eps)
                    print(tile_eps)
                    tile_segment_points = lidar_TilesubsetArr[row][col].iloc[:,:3].to_numpy()
                    subTileTree_Points,  _ = TileObj_MR.Classify_MultipleReturns(tile_segment_points,tile_eps)

                    

                    for t in subTileTree_Points:
                        Trees_Buffer.append(t)

                    if len(subTileTree_Points) > 0:
                        db = DBSCAN(eps=30, min_samples=30).fit(subTileTree_Points)
                        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                        core_samples_mask[db.core_sample_indices_] = True
                        DB_labels_t = db.labels_

                        for l in DB_labels_t:
                            DB_labels.append(l)
                    
                    logging.info("MR - T_ID : %s - ACTION: Trees Added to - S_ID : %d",filename,Tilecounter)

                else:
                    logging.warn("Empty Tileset Found")

        Trees_Buffer = np.array(Trees_Buffer)

        MR_TreesDf = pd.DataFrame(Trees_Buffer, columns=["X","Y","Z"])
        MR_TreesDf["class"] = DB_labels

        #Total Trees = number of unique classes
        Total_Trees = len(MR_TreesDf["class"].unique())
        logging.info("MR - T_ID : %s - ACTION: Total Trees : %d",filename,Total_Trees)

        #write it to the app in bold

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
        NG_df["class"] = -2
        Combined_df = pd.concat([GP_df, MR_TreesDf], ignore_index=True)

        # Add a download button to download the dataframe

        if len(Combined_df) > 0:
            
            st.download_button(
                label="Download classified data as CSV",
                data=Combined_df.to_csv(index=False),
                file_name='Classified_lidarData.csv',
                mime='text/csv',
                )

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

        # color_map = {}
        # for i in range(len(Combined_df["class"].unique())):
        #     color_map[str(i)] = px.colors.qualitative.Plotly[i]

        #     """
        #     color_map = {
        #             "0":"#636EFA"
        #             "1":"#EF553B"
        #             "2":"#00CC96"
        #             "3":"#AB63FA"
        #             "4":"#FFA15A"
        #     }
        #     """

        color_map = {
        "0": "#636EFA",
        "1": "#EF553B",
        "2": "#00CC96",
        "3": "#AB63FA",
        "4": "#FFA15A",
        "5": "#19D3F3",
        "6": "#FF6692",
        "7": "#B6E880",
        "8": "#FF97FF",
        "9": "#FECB52",
        "10": "#7F7F7F",
        "11": "#E64D66",
        "12": "#4DB380",
        "13": "#FF4D4D",
        "14": "#3C3C3C",
        "15": "#FF7F0E",
        "16": "#2CA02C",
        "17": "#D62728",
        "18": "#9467BD",
        "19": "#8C564B",
        "20": "#E377C2",
        "21": "#7F7F7F",
        "22": "#BCBD22",
        "23": "#17BECF",
        "24": "#9EDAE5",
        "25": "#FF9896",
        "26": "#98DF8A",
        "27": "#C5B0D5",
        "28": "#C49C94",
        "29": "#F7B6D2",
        "30": "#C7C7C7",
        "31": "#DBDB8D",
        "32": "#9EDAE5",
        "33": "#FF9896",
        "34": "#98DF8A",

        }

        #st.write(color_map)

        #For every Class in the Combined_df, assign a color from the color_map
        for i in range(len(Combined_df["class"].unique())):
            Combined_df["class"] = Combined_df["class"].replace(str(i),color_map[str(i)])#preventing overflow

        #Plot the combined dataframe
        fig = px.scatter_3d(Combined_df, x='X', y='Y', z='Z',
                        color='class', color_discrete_map=color_map)
        fig.update_traces(marker=dict(size=1.2))
        fig.update_layout(scene=dict(aspectmode='data'), showlegend=False)

        st.plotly_chart(fig)

        #Write the combined dataframe if user wants to see it
        if st.checkbox("Show Combined Dataframe"):
            st.write("Combined data:", Combined_df)
