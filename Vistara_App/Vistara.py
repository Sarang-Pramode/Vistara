import streamlit as st

import pandas as pd
import numpy as np
import laspy
import plotly.express as px

st.set_page_config(
        page_title="Visualizing LiDAR Point Clouds",
)


st.markdown("<h1 style='text-align: center; color: white;'>Project Vistara</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: \
            center; color: white;'> \
            A Streamlit App for Visualizing LiDAR Point Clouds</h3>"
            , unsafe_allow_html=True)

st.session_state["filename"] = "Vistara"

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

    #sample lidar_df
    lidar_df = lidar_df[::400]


    return lidar_df

filepath = "../lasFile_Reconstructed_25192.las"

# Process the uploaded file
point_cloud_df = process_las_file(filepath)

#Convert classification types to string
point_cloud_df["classification"] = point_cloud_df["classification"].astype(str)

# Create a color map for the classification types

color_map = {
    '1': 'white',
    '2': 'red',
    '4': 'green',
    '5': 'green',
    '6': 'green'
}

# Create 3D scatter plot using Plotly with classification-based colors
fig = px.scatter_3d(point_cloud_df, x='X', y='Y', z='Z', color='classification',
                    color_discrete_map=color_map,
                    hover_data=['X', 'Y', 'Z', 'classification'],
                    width=800, height=800)

fig.update_traces(marker=dict(size=1.2))
fig.update_layout(scene=dict(aspectmode='data'))

st.plotly_chart(fig)