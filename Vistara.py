import streamlit as st

import pandas as pd
import numpy as np
import laspy
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit.source_util import get_pages


st.set_page_config(
        page_title="Visualizing LiDAR Point Clouds",
        layout='wide'
)

from streamlit.components.v1 import html

def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)

st.markdown("<h1 style='text-align: center; color: white;'>Vistara</h1>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: \
            center; color: white;'> \
            LiDAR Point Cloud Classification Toolkit for Sustainable Urban Development</h3>"
            , unsafe_allow_html=True)

st.session_state["filename"] = "Vistara"
st.session_state["Extracted_Lidar_Data"] = None
st.session_state["boxSize"] = 100

#Add a gap between the title and the rest of the page

st.markdown("<br>", unsafe_allow_html=True)


# Define the benefits of the toolkit
benefits = [
    {
        "title": "Real Time Point Cloud Classification ",
        "description": "Our toolkit simplifies the process of adding classification labels to unstructured point cloud data obtained from LiDAR scans, making it easy for users to identify and classify vegetation, ground, and building points in urban landscapes.",
    },
    {
        "title": "Extract Accurate Canopy Metrics",
        "description": "By extracting high-quality canopy metrics and tree characteristics, our toolkit enables users to make equitable tree planting decisions and analyze the impact of trees as an infrastructure to a neighborhood, fostering more sustainable urban development.",
        
    },
    {
        "title": "Urban Features from 3d Point Clouds",
        "description": "Our approach uses unsupervised learning techniques to identify and classify features in LiDAR scans, eliminating the need for manual labeling and saving time and resources.",
        
    },
    {
        "title": "Visualize and Validate Results",
        "description": "Our pipeline is hosted with Streamlit, providing a user-friendly interface for selecting a region on a map, downloading the raw data, and visualizing the results",
        
    }
]

# Load the images from file
image1 = Image.open("images/blank cloud.png")
image2 = Image.open("images/tree_shadef.png")
image3 = Image.open("images/ground_E.png")
image4 = Image.open("images/labelled_points.png")

img_size = (420,300)

# center and resize the images
image1 = image1.resize(img_size, resample=Image.LANCZOS)
image2 = image2.resize(img_size, resample=Image.LANCZOS)
image3 = image3.resize(img_size, resample=Image.LANCZOS)
image4 = image4.resize((430,300), resample=Image.LANCZOS)

# add a black border to the image
border_width = 50
border_image1 = ImageOps.expand(image1, border=border_width, fill='black')
border_image2 = ImageOps.expand(image2, border=border_width, fill='black')
border_image3 = ImageOps.expand(image3, border=border_width, fill='black')
border_image4 = ImageOps.expand(image4, border=border_width, fill='black')


Fontsize = 20
# Define the layout of the cells using a 2x2 grid
col1, col2 = st.columns(2)

# Define the contents of the cells with rounded edges and a border
with col1:

    # Define the text to add to the image
    text_img1 = benefits[0]['title']

    # Create a new image with the text added
    draw = ImageDraw.Draw(border_image1)
    font = ImageFont.truetype("open-sans/OpenSans-Bold.ttf", Fontsize)
    textwidth, textheight = draw.textsize(text_img1, font)
    x = (border_image1.width - textwidth) / 2
    y = border_image1.height - Fontsize/0.5 #(image1.height - textheight) / 18
    draw.text((x, y), text_img1, fill="white", font=font)

    st.image(border_image1, width=400, use_column_width=True, caption=benefits[0]['description'])

with col2:

    # Define the text to add to the image
    text_img2 = benefits[1]['title']

    # Create a new image with the text added
    draw = ImageDraw.Draw(border_image2)
    font = ImageFont.truetype("open-sans/OpenSans-Bold.ttf", Fontsize)
    textwidth, textheight = draw.textsize(text_img2, font)
    x = (border_image2.width - textwidth) / 2
    y = border_image2.height - Fontsize/0.5 #(image1.height - textheight) / 18
    draw.text((x, y), text_img2, fill="white", font=font)

    st.image(border_image2, width=400, use_column_width=True, caption=benefits[1]['description'])

    
col3, col4 = st.columns(2)

with col3:
    text_img3 = benefits[2]['title']

    # Create a new image with the text added
    draw = ImageDraw.Draw(border_image3)
    font = ImageFont.truetype("open-sans/OpenSans-Bold.ttf", Fontsize)
    textwidth, textheight = draw.textsize(text_img3, font)
    x = (border_image3.width - textwidth) / 2
    y = border_image3.height - Fontsize/0.5 #(image1.height - textheight) / 18
    draw.text((x, y), text_img3, fill="white", font=font)

    st.image(border_image3, width=400, use_column_width=True, caption=benefits[2]['description'])

with col4:
    text_img4 = benefits[3]['title']

    # Create a new image with the text added
    draw = ImageDraw.Draw(border_image4)
    font = ImageFont.truetype("open-sans/OpenSans-Bold.ttf", Fontsize)
    textwidth, textheight = draw.textsize(text_img4, font)
    x = (border_image4.width - textwidth) / 2
    y = border_image4.height - Fontsize/0.5 #(image1.height - textheight) / 18
    draw.text((x, y), text_img4, fill="white", font=font)

    st.image(border_image4, width=400, use_column_width=True, caption=benefits[3]['description'])

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

#Add a space of 10 pixels between the columns
st.markdown("<style>.main * div:nth-child(2) > div{padding-top: 30px;}</style>", unsafe_allow_html=True)

# Add a horizontal line that separates the footer from the main content and spans the entire width of the app
st.markdown("<hr style='border: 1px solid grey;'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:


    # Add a title to the app
    st.markdown("<h1 style='text-align: center; color: white;'>About</h1>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: \
                left; color: white;'> \
                A large number of cities have been conducting LiDAR surveys to better map their cities. \
                Most of these surveys are public but requires significant technical expertise to extract and analyze. \
                To aid in the analysis of the neighbourhoods through LiDAR, I built this app to showcase the capabilities of an open source python package \
                I call <span style='color: #08a308; font-weight:bold;'>TerraVide</span>. You can check it out <a href='https://pypi.org/project/TerraVide/' target='_blank' style='color: #5f87c7;'>here</a><p> \
                <br> \
                Vistara was built to showcase some capabilites of <span style='color: #08a308; font-weight:bold;'>TerraVide</span>. \
                The tile you see on the right represents the classification of over <span style= font-weight:bold;'>12 Million</span> raw X,Y,Z coordinates into  \
                <span style= font-weight:bold;'>Trees</span>,\
                <span style= font-weight:bold;'>Buildings</span>, \
                and <span style= font-weight:bold;'>Ground</span></p>"
                , unsafe_allow_html=True)

    # # Write the text - you can check out my repo here
    # st.markdown("<h6 style='text-align: \
    #             left; color: grey;'> \
    #             You can check out TerraVide (Its Open Source!) <a href='https://pypi.org/project/TerraVide/' target='_blank' style='color: #5f87c7;'>here</a></h6>"
    #             , unsafe_allow_html=True)


    # # Add a small footer to the end of the streamlit app with the author's name
    # st.markdown("<h6 style='text-align: \
    #             left; color: grey;'> \
    #             Let me know what you think of this app! Connect with me on <a href='https://www.linkedin.com/in/sarang-pramode-713b99167/' style='color: #5f87c7;' >LinkedIn! </a></h6> \
    #             <h6 style='text-align: \
    #             left; color: grey;'> \
    #             Made by <a href='https://www.sarangpramode.com/' style='color: #5f87c7;' >Sarang Pramode</a></h6>"
    #             , unsafe_allow_html=True)

with col2:

    filepath = "lasFile_Reconstructed_25192_sampled.las"

    # Process the uploaded file
    point_cloud_df = process_las_file(filepath, sampling_rate=15)

    #Convert classification types to string
    point_cloud_df["classification"] = point_cloud_df["classification"].astype(str)

    # Create a color map for the classification types

    color_map = {
        '1': 'white',
        '2': 'red',
        '4': 'blue',
        '5': 'green',
        '6': 'green'
    }

    # Create 3D scatter plot using Plotly with classification-based colors
    fig = px.scatter_3d(point_cloud_df, x='X', y='Y', z='Z', color='classification',
                        color_discrete_map=color_map,
                        hover_data=['X', 'Y', 'Z', 'classification'],
                        width=300, height=400,
                        opacity=0.8)

    fig.update_traces(marker=dict(size=1.2))
    fig.update_layout(scene=dict(aspectmode='data'), showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Set Zoom for the plot
    fig.update_layout(scene_camera_eye=dict(x=1, y=1, z=1))

    st.plotly_chart(fig, use_container_width=True)

# Ask the user to use the sidebar to go to the next page
st.markdown("<h6 style='text-align: \
            left; color: grey;'> \
            Use the sidebar to go to the next page</h6>"
            , unsafe_allow_html=True)


# Add a horizontal line that separates the footer from the main content and spans the entire width of the app
st.markdown("<hr style='border: 1px solid grey;'>", unsafe_allow_html=True)

#Center align the text to the middle of the column
st.markdown("<p style='text-align: center; color: white;'>Built with ❤️ by <a href='https://www.sarangpramode.com/' target='_blank' style='color: #5f87c7;'>Sarang Pramode</a></p>", unsafe_allow_html=True)  

# Add a version number to the footer
st.markdown("<h6 style='text-align: \
            left; color: #2d3645;'> \
            Version 1.2.1</h6>"
            , unsafe_allow_html=True)

# #Dont show any axis values in the plot
# fig.update_layout(scene = dict(
#                     xaxis = dict(
#                         backgroundcolor="rgb(255, 255, 255)",
#                         gridcolor="rgb(255, 255, 255)",
#                         showbackground=True,
#                         zerolinecolor="rgb(255, 255, 255)",
#                         showticklabels=False,
                        
#                         ),

#                     yaxis = dict(
                        
#                         backgroundcolor="rgb(255, 255,255)",
#                         gridcolor="rgb(255, 255, 255)",
#                         showbackground=True,
#                         zerolinecolor="rgb(255, 255, 255)",
#                         showticklabels=False
#                         ),
#                     zaxis = dict(
                        
#                         backgroundcolor="rgb(255, 255,255)",
#                         gridcolor="rgb(255, 255, 255)",
#                         showbackground=True,
#                         zerolinecolor="rgb(255, 255, 255)",
#                         showticklabels=False)
#                         )
                
#                 )