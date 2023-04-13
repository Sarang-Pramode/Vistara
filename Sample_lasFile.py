import laspy
import pandas as pd
import numpy as np
import os


# Read the las file
# Function to process the LAS file and convert it to a DataFrame
def input(filepath, sample=400):
    
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
    lidar_df = lidar_df[::sample]


    return lidar_df

def output(lidar_df, filename):

    las = laspy.create(file_version="1.4", point_format=3)

    Xscale = 0.01
    Yscale = 0.01
    Zscale = 0.01
    Xoffset = 0
    Yoffset = 0
    Zoffset = 0

    las.header.offsets = [Xoffset,Yoffset,Zoffset]
    las.header.scales = [Xscale,Yscale,Zscale]
    las.x = lidar_df["X"].to_numpy()
    las.y = lidar_df["Y"].to_numpy()
    las.z = lidar_df["Z"].to_numpy()
    las.intensity = [0]*len(lidar_df)
    las.classification =  lidar_df["classification"].to_numpy()
    las.return_number = lidar_df["return_number"].to_numpy()
    las.number_of_returns = lidar_df["number_of_returns"].to_numpy()

    las.write(filename)

if __name__ == "__main__":

    filepath = "lasFile_Reconstructed_25192.las"

    # Process the uploaded file
    point_cloud_df = input(filepath)

    #Write the output file
    output(point_cloud_df, "lasFile_Reconstructed_25192_sampled.las")



