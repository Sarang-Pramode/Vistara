import streamlit as st
import pandas as pd
import numpy as np
import laspy
import plotly.express as px

# Set up Streamlit app
st.title("LiDAR Point Cloud Visualization")
st.sidebar.title("LiDAR Data Upload")

def Create_lasFileDataframe(lasfileObject):
    """Take a lasfile object after reading <filename>.las and convert into a Pandas Dataframe
    Columns Stored = {'X','Y','Z','return_number','number_of_returns'}
    Coordinates are in ft 

    Args:
        lasfileObject (_type_): lasfile Object after running Read_lasFile(filepath) function

    Returns:
        lidar_Dataframe: Pandas Dataframe of lidar points as well as return number and number of returns for each data point
    """

    #Making a datframe from the lidar data
    Xscale = lasfileObject.header.x_scale
    Yscale = lasfileObject.header.y_scale
    Zscale = lasfileObject.header.z_scale

    Xoffset = lasfileObject.header.x_offset
    Yoffset = lasfileObject.header.y_offset
    Zoffset = lasfileObject.header.z_offset

    lidarPoints = np.array(
        ( 
        (lasfileObject.X*Xscale)/3.28 + Xoffset,  # convert ft to m and correct measurement
        (lasfileObject.Y*Yscale)/3.28 + Yoffset,
        (lasfileObject.Z*Zscale)/3.28 + Zoffset,
        lasfileObject.classification,
        lasfileObject.return_number, 
        lasfileObject.number_of_returns)).transpose()
    lidar_df = pd.DataFrame(lidarPoints , columns=['X','Y','Z','classification','return_number','number_of_returns'])

    #Filtering out noise
    lidar_df = lidar_df[lidar_df["classification"] != 18] #removing high noise
    lidar_df = lidar_df[lidar_df["classification"] != 7] #removing  noise

    #Raw point cloud data
    rawPoints = np.array((
        ((lidar_df.X)*(Xscale)) + Xoffset, # convert ft to m
        (lidar_df.Y)*(Yscale) + Yoffset, #convert ft to m
        (lidar_df.Z)*(Zscale) + Zoffset
    )).transpose()

    return lidar_df, rawPoints

# Function to process the LAS file and convert it to a DataFrame
def process_las_file(filepath):
    
    las_file = laspy.read(filepath)

    #Making a datframe from the lidar data
    Xscale = las_file.header.x_scale
    Yscale = las_file.header.y_scale
    Zscale = las_file.header.z_scale

    Xoffset = las_file.header.x_offset
    Yoffset = las_file.header.y_offset
    Zoffset = las_file.header.z_offset

    lidarPoints = np.array(
        ( 
        (las_file.X*Xscale)/3.28 + Xoffset,  # convert ft to m and correct measurement
        (las_file.Y*Yscale)/3.28 + Yoffset,
        (las_file.Z*Zscale)/3.28 + Zoffset,
        las_file.classification,
        las_file.return_number, 
        las_file.number_of_returns)).transpose()
    lidar_df = pd.DataFrame(lidarPoints , columns=['X','Y','Z','classification','return_number','number_of_returns'])

    #change classification to int
    lidar_df["classification"] = lidar_df["classification"].astype(int)

    # #Raw point cloud data
    # rawPoints = np.array((
    #     ((lidar_df.X)*(Xscale)) + Xoffset, # convert ft to m
    #     (lidar_df.Y)*(Yscale) + Yoffset, #convert ft to m
    #     (lidar_df.Z)*(Zscale) + Zoffset
    # )).transpose()


    #points = np.vstack((las_file.x, las_file.y, las_file.z)).T
    
    # Sample the points
    # points = rawPoints[::200]
    #classifications = las_file.classification[::200].astype(np.uint8)

    #sample lidar_df
    lidar_df = lidar_df[::200]
    # classifications = lidar_df.classification

    # point_cloud_df = pd.DataFrame(points, columns=["x", "y", "z"])
    # point_cloud_df["classification"] = classifications

    return lidar_df

#filepath = "lasFile_Reconstructed_25192.las"
filepath = "/Volumes/Elements/TerraVide/Datasets/EC2_data/Package_Generated/25192/2017/LasClassified_25192/lasFile_Reconstructed_25192.las"
# Process the uploaded file
point_cloud_df = process_las_file(filepath)

# Create 3D scatter plot using Plotly
# fig = px.scatter_3d(point_cloud_df, x='x', y='y', z='z', color='z',
#                     color_continuous_scale=px.colors.sequential.Viridis,
#                     range_color=[point_cloud_df['z'].min(), point_cloud_df['z'].max()],
#                     hover_data=['x', 'y', 'z'])

# Custom color map for six classification values
color_map = {
    1: 'red',
    2: 'blue',
    3: 'green',
    4: 'yellow',
    5: 'purple',
    6: 'orange',
}


# Create 3D scatter plot using Plotly with classification-based colors
fig = px.scatter_3d(point_cloud_df, x='X', y='Y', z='Z', color='classification',
                    color_discrete_map=color_map,
                    hover_data=['X', 'Y', 'Z', 'classification'])


fig.update_traces(marker=dict(size=1, opacity=0.7))
fig.update_layout(scene=dict(aspectmode='data'))

st.plotly_chart(fig)
st.write("Point cloud data:", point_cloud_df)

