import streamlit as st

import pandas as pd
import numpy as np
import laspy
import plotly.express as px

st.set_page_config(
        page_title="Visualizing LiDAR Point Clouds",
        initial_sidebar_state='collapsed',
        layout='wide'
)


st.markdown("<h1 style='text-align: center; color: white;'>Project Vistara</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: \
            center; color: white;'> \
            A Streamlit App for Visualizing LiDAR Point Clouds</h3>"
            , unsafe_allow_html=True)

st.session_state["filename"] = "Vistara"

# Function to process the LAS file and convert it to a DataFrame
def process_las_file(filepath, sampling_rate=10):
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
    lidar_df = lidar_df[::sampling_rate]


    return lidar_df

filepath = "lasFile_Reconstructed_25192_sampled.las"

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
                    width=400, height=500)

fig.update_traces(marker=dict(size=1.2))
fig.update_layout(scene=dict(aspectmode='data'), showlegend=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)


st.plotly_chart(fig, use_container_width=True)

st.markdown("<h4 style='text-align: \
            center; color: white;'> \
            Vistara was built to showcase the capabilites of TerraVide \
            - an open source python package to process LiDAR data <br> \
            The Tile you see above was classified into Trees , Buildings and ground from raw X,Y,Z coordinates</h4>"
            , unsafe_allow_html=True)

# Add a horizontal line that separates the footer from the main content and spans the entire width of the app
st.markdown("<hr style='border: 1px solid grey;'>", unsafe_allow_html=True)

#Add a link to the TerraVide GitHub repo
st.markdown("<h6 style='text-align: \
            left; color: grey;'> \
            Built with <a href='https://pypi.org/project/TerraVide/' target='_blank'>TerraVide</a></h4>"
            , unsafe_allow_html=True)

# Add a small footer to the end of the streamlit app with the author's name
st.markdown("<h6 style='text-align: \
            left; color: grey;'> \
            Built by <a href='https://www.linkedin.com/in/sarang-pramode-713b99167/'>Sarang Pramode</a></h4>"
            , unsafe_allow_html=True)
