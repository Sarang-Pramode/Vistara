import streamlit as st

# This page describes what LiDAR is and how it works and gives a brief overview of the project

st.markdown("<h2>What is LiDAR?</h2>", unsafe_allow_html=True)

#Add a line break
st.markdown("<br>", unsafe_allow_html=True)

#Add a paragraph
st.markdown("""LiDAR is a technology that uses lasers to measure distances and create precise 3D maps of the environment. The word "LiDAR" stands for "Light Detection and Ranging", and it is pronounced "li-DAR".""")

"""
The basic principle of LiDAR is simple: a laser sends out pulses of light, which bounce off objects in the environment and return to the LiDAR sensor. By measuring the time it takes for the light to travel back, the LiDAR sensor can calculate the distance to the object.
"""

#Add an image and align it to the center
st.image("https://maps.nyc.gov/lidar/2017/img/topo.png", caption="LiDAR point cloud Data")

"""
This process is repeated many times per second, generating a dense point cloud of millions of 3D points that represent the shape and location of objects in the environment. This point cloud can be used for various applications, such as surveying, mapping, and object detection.
"""

st.markdown("<h3>How does LiDAR work?</h3>", unsafe_allow_html=True)

""" 
LiDAR sensors are typically mounted on vehicles, drones, or other platforms that move through the environment. The LiDAR sensor emits pulses of laser light in a particular direction, and measures the time it takes for the light to reflect off objects and return to the sensor. By combining this information with the location and orientation of the sensor, the LiDAR system can create a 3D map of the environment.
LiDAR sensors can emit laser pulses in different patterns and frequencies, depending on the specific application. For example, some LiDAR systems use multiple lasers to scan the environment from different angles, while others use a single laser that rotates rapidly to cover a wide field of view.
"""

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("LiDAR scan example.png", width=300, caption="LiDAR scan example")

st.markdown("<h3>What are the applications of LiDAR?</h3>", unsafe_allow_html=True)

#Add list of bullet points


"""
LiDAR has many applications in various fields, such as:
"""

#Add markdown text which is formatted as a list



st.markdown(""" \
    <ul>
        <li><b>Surveying and mapping:</b> LiDAR can be used to create highly accurate 3D maps of terrain, buildings, and other objects. This is useful for urban planning, construction, and environmental monitoring.</li>
        <li><b>Autonomous vehicles:</b> LiDAR is a key technology for self-driving cars, as it can provide real-time information about the environment and help the vehicle navigate safely.</li>
        <li><b>Forestry and agriculture:</b> LiDAR can be used to measure tree height, density, and biomass, which is useful for forest management and carbon sequestration. It can also be used to optimize crop yields by measuring plant height, spacing, and health.</li>
        <li><b>Environmental monitoring:</b> LiDAR can be used to monitor changes in the environment, such as coastal erosion, glacier melting, and land use changes.</li>
        <li><b>Object detection and tracking:</b> LiDAR can be used to detect and track objects in the environment, such as pedestrians, vehicles, and wildlife.</li>
    </ul>
""", unsafe_allow_html=True)


# Add a horizontal line that separates the footer from the main content and spans the entire width of the app
st.markdown("<hr style='border: 1px solid grey;'>", unsafe_allow_html=True)


