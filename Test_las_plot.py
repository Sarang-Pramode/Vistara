import pyvista as pv
import laspy
import numpy as np

# Load LAS file
las_data = laspy.read("lasFile_Reconstructed_25192.las")

# Extract point cloud data
points = las_data.points
x, y, z = points['X'], points['Y'], points['Z']

# Create PyVista point cloud object
point_cloud = pv.PolyData(np.vstack((x, y, z)).transpose())

# Create PyVista plotter
plotter = pv.Plotter()

# Add point cloud to plotter
plotter.add_mesh(point_cloud, color='white', point_size=1.0)

# Set plotter background color
plotter.background_color = 'black'

# Display plotter window
plotter.show(interactive=True)
