"""
This script data_utils.py contains all the functions to be used in the main of
folder coords_transform.
"""


from objects_localization.coords_transform.common.Device import Streaming
from objects_localization.coords_transform.common import Data

import math
import numpy as np
import logging
import cv2


#############################################################
#######    Get the World Coords and Distance Data     #######
#############################################################
def convert2cartesian_pixelwise(dist_data,
                                ints_data,
                                cnfi_data,
                                cam_params,
                                is_stereo):
    """
    The convert2cartesian_pixelwise function takes in the following parameters:
        dist_data: The distance data from the camera, reshaped to a 2D array.
        ints_data: The intensity data from the camera, reshaped to a 2D array.
        cnfi_data: The confidence information (0 or 1) for each pixel of
        ints_data and dist_data, reshaped to a 2D array.
    
    Args:
        dist_data: Define the output data type
        ints_data: Store the intensity data
        cnfi_data: Determine which points are valid
        cam_params: Pass the camera parameters to the
                       convert2cartesian_pixelwise function
        is_stereo: Distinguish between the two cameras
    
    Returns:
        world_coords: Nested List with the line wise data. Each list item is a
                      list with the following entries X Y Z R G B I i.e.
                      point coordinates (XYZ), color (RGB) and intensity(I).
        dist_data: input dist_data reshaped to array with camera resolution
    
    Doc Author:
        Afonso Santos, Viridius Technology
    """

    world_coords = []

    m_c2w = np.array(cam_params.cam2worldMatrix)
    shape = (4, 4)
    m_c2w.shape = shape


    cnfi_data = np.asarray(cnfi_data).reshape(cam_params.height,
                                              cam_params.width)
    ints_data = np.asarray(ints_data).reshape(cam_params.height,
                                              cam_params.width)
    dist_data = np.asarray(list(dist_data)).reshape(cam_params.height,
                                                    cam_params.width)

    if is_stereo:
        #RGBA intensities
        ints_data = np.asarray(ints_data).astype('uint32').view('uint8').reshape(cam_params.height,
                                                                                 cam_params.width,
                                                                                 4)
        ints_data = np.frombuffer(ints_data, np.uint8).reshape(cam_params.height,
                                                               cam_params.width,
                                                               4)
        color_map = ints_data

        # Apply the Statemap to the Z-map
        zmapData_with_statemap = np.array(dist_data).reshape(cam_params.height,
                                                             cam_params.width)

        for row in range(cam_params.height):
            for col in range(cam_params.width):
                # if (cnfi_data[row][col] != 0):
                #     zmapData_with_statemap[row][col] = 0 # Set invalid pixels to lowest value
                # else:
                #     # use all "good" points to export to PLY

                # transform into camera coordinates (zc, xc, yc)
                xp = (cam_params.cx - col) / cam_params.fx
                yp = (cam_params.cy - row) / cam_params.fy

                # coordinate system local to the imager
                zc = dist_data[row][col]
                xc = xp * zc
                yc = yp * zc

                # Convert to world coordinate system
                xw = (m_c2w[0, 3] + zc * m_c2w[0, 2] + yc * m_c2w[0, 1] + xc * m_c2w[0, 0])
                yw = (m_c2w[1, 3] + zc * m_c2w[1, 2] + yc * m_c2w[1, 1] + xc * m_c2w[1, 0])
                zw = (m_c2w[2, 3] + zc * m_c2w[2, 2] + yc * m_c2w[2, 1] + xc * m_c2w[2, 0])

                # Merge 3D coordinates and color
                world_coords.append([xw,
                                     yw,
                                     zw,
                                     color_map[row][col][0],
                                     color_map[row][col][1],
                                     color_map[row][col][2],
                                     0])

        return world_coords, dist_data

    else:
        for row in range(0, cam_params.height):
            for col in range(0, cam_params.width):

                #calculate radial distortion
                xp = (cam_params.cx - col) / cam_params.fx
                yp = (cam_params.cy - row) / cam_params.fy

                r2 = (xp * xp + yp * yp)
                r4 = r2 * r2

                k = 1 + cam_params.k1 * r2 + cam_params.k2 * r4

                xd = xp * k
                yd = yp * k

                d = dist_data[row][col]
                s0 = np.sqrt(xd*xd + yd*yd + 1)

                xc = xd * d / s0
                yc = yd * d / s0
                zc = d / s0 - cam_params.f2rc

                # convert to world coordinate system
                xw = (m_c2w[0, 3] + zc * m_c2w[0, 2] + yc * m_c2w[0, 1] + xc * m_c2w[0, 0])
                yw = (m_c2w[1, 3] + zc * m_c2w[1, 2] + yc * m_c2w[1, 1] + xc * m_c2w[1, 0])
                zw = (m_c2w[2, 3] + zc * m_c2w[2, 2] + yc * m_c2w[2, 1] + xc * m_c2w[2, 0])

                # convert to full decibel values * 0.01, which is the same format that Sopas uses for point cloud export
                intsSopasFormat = round(0.2 * math.log10(ints_data[row][col]), 2) if ints_data[row][col] > 0 else 0

                # merge 3D coordinates and intensity
                world_coords.append([xw, yw, zw, 0, 0, 0, intsSopasFormat])

        return world_coords, dist_data


#############################################################
#######    Get the Frame Data from V3SCam Device      #######
#############################################################
def get_frame_data(ip_addr, tcp_port):
    """
    The get_frame_data function reads the data from a frame and returns it as a
    numpy array. The function also converts the depth map to world coordinates,
    which are returned in an array.

    Args:
        ip_addr: Specify the ip address of the device
        tcp_port: Specify the port number to be used for

    Returns:
        The world coordinates and the distance data of a single frame

    Doc Author:
        Afonso Santos, Viridius Technology
    """

    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.WARNING)
    device_streaming = Streaming(ip_addr, tcp_port)
    device_streaming.openStream()
    device_streaming.sendBlobRequest()

    # Variable to control when stop the capture of frames
    dont_stop = True

    # Data -> Data Class
    raw_data = Data.Data()

    # Variable to control the number of frames to capture
    i = 0
    while dont_stop:
        try:
            # One frame only
            if i == 0:
                dont_stop = False

            # Get the whole frame
            device_streaming.getFrame()
            frame = device_streaming.frame

            raw_data.read(frame)

            if raw_data.hasDepthMap:
                print('\nData contains depth map data:')
                print('Frame number: ', raw_data.depthmap.frameNumber)

                distanceData = raw_data.depthmap.distance

                numCols = raw_data.cameraParams.width
                numRows = raw_data.cameraParams.height

                # Convert to World Coordinates
                world_coords, dist_data = \
                    convert2cartesian_pixelwise(raw_data.depthmap.distance,
                                                raw_data.depthmap.intensity,
                                                raw_data.depthmap.confidence,
                                                raw_data.cameraParams,
                                                raw_data.xmlParser.stereo)
            # Increment the variable to control the number of frames captured
            i = i + 1

        except KeyboardInterrupt:
            print('Ctrl-C pressed (1), terminating.')
            dont_stop = False

    device_streaming.closeStream()

    return world_coords, dist_data


#############################################################
#######    Visualize the Matrix with World Coords     #######
#############################################################
def get_matrix(world_coords):
    """
    The get_matrix function takes in the world coordinates of an image and
    returns a matrix with 6 layers. The first layer is the x-coordinates,
    second layer is y-coordinates, third layer is z-coordinates, fourth layer
    is red values, fifth layer is green values and sixth layer are blue values.
    The function iterates over the rows and columns of the image to fill each
    value in its respective location.

    Args:
        world_coords: Get the coordinates of each point in the image

    Returns:
        A matrix of size (512, 640, 6)

    Doc Author:
        Afonso Santos, Viridius Technology
    """

    matrix = np.zeros((512, 640, 6))

    # This cycle iterates over the 6 layers of the image
    for layer in range(len(world_coords[0])-1):
        index = 0
        # This cycles iterates over the rows and cols and fill them
        for row in range(512):
            for col in range(640):
                matrix[row][col][layer] = world_coords[index][layer]
                index += 1

    return matrix


###################### FOR VISUAL TEST ######################

#############################################################
#########    Visualize the Frame using the cv2     ##########
#############################################################
def show_frame(matrix, weight_rgb, weight_depth):
    """
    The show_frame function takes in a matrix of RGB and depth values, as well
    as two weights. It then combines the images together using the
    cv2.addWeighted function, with the RGB image having a weight of w_rgb and
    the depth image having a weight of w_depth. It also rotates this combined
    image 90º.

    Args:
        matrix: Pass the image data from the read_frame function to show_frame
        weight_rgb: Control the balance between rgb and depth image
        weight_depth: Adjust the depth image's brightness

    Returns:
        The rgb and depth images together

    Doc Author:
        Afonso Santos, Viridius Technology
    """

    # Get the x, y, and z depth values from the first 3 layers of the image
    x = matrix[:, :, 0]
    y = matrix[:, :, 1]
    z = matrix[:, :, 2]

    # Normalize the depth values to be between 0 and 1
    depth = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    depth = depth / depth.max()

    # Create a blue-to-red color map for the depth values
    depth = depth[:, :, np.newaxis]
    depth = np.concatenate((np.zeros_like(depth), depth, depth), axis=2)

    # Get the RGB values from the last 3 layers of the image
    rgb = matrix[:, :, 3:]

    # Add the RGB image and depth image together
    frame_img = cv2.addWeighted(rgb, weight_rgb, depth, weight_depth, 0)
    # Rotate 90º
    #frame_img = cv2.rotate(frame_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('RGB and Depth Images Together', frame_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
