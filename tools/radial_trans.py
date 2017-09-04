import numpy as np
import scipy.ndimage as ndi

def radial_transform(img, polar = None, 
        row_axis=1, col_axis=2, channel_axis=0):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        img: Input tensor. Must be 3D.
        polar: The pole in the polar coordinate system.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
    # Returns
        Rotated Numpy image tensor.
    """
    new_image = np.zeros(img.shape, dtype = type(img[0,0,0]))
    height, width = img.shape[row_axis], img.shape[col_axis]
    if img.shape[channel_axis] > 3:
        print "Warning: number of channels in the input image tensor > 3"

    if polar == None:
        m, n = height / 2, width / 2
    else:
        m, n = polar
    
    for u in range(height):
        theta_u = 2 * np.pi * u / height
        for v in range(width):
            x = int(v * np.cos(theta_u))
            y = int(v * np.sin(theta_u))
            if m + x >= 0 and m + x < height and n+y >= 0 and n+y < width:
                new_image[:, u, v] = img[:, m+x, n+y]

    return new_image

