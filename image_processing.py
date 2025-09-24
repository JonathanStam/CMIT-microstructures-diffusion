import cv2
import torch
import numpy as np
from skimage.feature import peak_local_max
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.draw import line
from skimage.segmentation import watershed

def Sobel_thres(img: torch.tensor, thres=26):
    """
    Perform Sobel edge detection and then threshold the result to get a binary mask. To match pytorch format, img given in CxHxW format
    """
    img = img.transpose(0,-1).numpy()

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
 
    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    
    threshholded_sob = gradient_magnitude > thres

    return threshholded_sob

from medpy.filter.smoothing import anisotropic_diffusion

def Sobel_thres_anis(img, thres=26):
    """
    Sobel thres but preprocess with an anisotropic diffusion kernel
    """
    img = img.transpose(0,-1).numpy()

    img = anisotropic_diffusion(img)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) 
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
 
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    
    threshholded_sob = gradient_magnitude > thres

    return threshholded_sob




# We code up the segmentation pipeline to get grain maps from our generated images:
# We work with numpy arrays in HxWxC format, so may want to start with img = img.transpose(0,-1).nunpy() with the original pytorch tensor.

def Sobel(img: np.array):
    """Get the edge map using Sobel edge detection"""

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
 
    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)

    return gradient_magnitude

def dist_map(img: np.array):
    """Takes an edge map and returns the map of distances from each point to the set of edge points"""
    img = np.uint8(img)
    img = 255*(1 - img)

    dist = cv2.distanceTransform(img, cv2.DIST_L2, maskSize=3)

    return dist

def mean_intensity(coord, img, radius=1):
    h, w = img.shape[0], img.shape[1]
    mean = 0
    n = 0
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            x = coord[0] + i
            y = coord[1] + j
            if 0 <= x < h and 0 <= y < w:
                n += 1
                mean = ((n-1)*mean + img[x][y])/n
    return mean

def edge_between(p1, p2, img, thres=20):
    """Determine if there is a high value of img along the line between p1 and p2"""
    l = np.array(line(p1[0],p1[1],p2[0],p2[1]))
    ncols = img.shape[1]
    #print(ncols)

    l_flat = ncols * l[0] + l[1]
    img_flat = img.flatten()

    values = img_flat[l_flat]
    #print(values)

    if values.max() > thres:
        return True
    else:
        return False
    

def get_maxes(img, min_distance=2):
    
    padded_img = cv2.copyMakeBorder(img, 5,5,5,5, borderType=cv2.BORDER_CONSTANT)
    maxes = peak_local_max(image=padded_img, min_distance=min_distance)
    maxes = maxes - 5

    return maxes


def segment(img: np.array, edge_detector=Sobel, edge_thres=26, smooth_dist=3):
    """"""
    edge_intensity = edge_detector(img)
    edge_mask = edge_intensity > edge_thres

    dist = dist_map(edge_mask)
    for i in range(smooth_dist):
        dist = anisotropic_diffusion(dist)

    maxes = get_maxes(dist)

    # Now we want to remove maxima that are in the same grain. We do that first by comparing the intensities of all the maxima to see an possible double counting, and then look whether an edge lies on the intervening line.
    sus_points = [] # This is to be a list of lists, where the ith element is the list of points further down the list than i which have similar intesity (so that we can evaluate them systematically later.)
    intensity_thres = 5
    for i in range(len(maxes)):
        sus_points.append([])
        for j in range(i+1, len(maxes)):
            max1 = maxes[i]
            max2 = maxes[j]
            if abs(mean_intensity(max1, img=img) - mean_intensity(max2, img=img)) < intensity_thres:
                sus_points[i].append(max2.tolist())

    destroy = []
    lines = []
    for i in range(len(maxes)):
        max1 = maxes[i]

        for max2 in sus_points[i]:
            lines.append(line(max1[0],max1[1],max2[0],max2[1]))
            if edge_between(max1, max2, edge_intensity, thres=20) == False and not(max2 in destroy):
                destroy.append(max2)


    culled_maxes = np.array([max for max in maxes.tolist() if max not in destroy])

    h, w = img.shape[:2]
    markers = np.zeros((h,w), dtype=np.int32)
    for i in range(len(culled_maxes)):
        r, c = culled_maxes[i]
        markers[r][c] = 1 + i


    segments = watershed(image=edge_intensity, markers=markers)

    return segments

