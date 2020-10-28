"""
This module contains functions to preprocess multispectral satellite images (creation
of RGB images through pixel rescaling / downsampling / pansharpening, cloud masks,
different remote sensing indices, etc.) once downloaded from the GEE server.
Note that these methods are meant to be a quick hack, without histogram correction
techniques or additional steps that would be required for a perfectly clean output.

Original author: Kilian Vos, Water Research Laboratory,
                 University of New South Wales, 2018
                 https://github.com/kvos/CoastSat
Modifications and additions: Thomas de Mareuil, Total E-LAB, 2020
"""

# load image processing modules
import numpy as np
import skimage
import skimage.transform as transform
import skimage.morphology as morphology
import sklearn.decomposition as decomposition
import skimage.exposure as exposure
from osgeo import gdal
from osgeo import osr
import matplotlib.pyplot as plt
import ee
from ee import batch

# additional modules
import os
import pdb
import shapely
import time

# other totalsat modules
from totalsat import sat_tools

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans


###################################################################################################
# RASTER PROCESSING
###################################################################################################

def create_cloud_mask(im_QA, satname, cloud_mask_issue):
    """
    Creates a cloud mask using the information contained in the Quality Assessment (QA) band
    for each satellite.

    Arguments:
    -----------
    im_QA: np.array
        Image containing the QA band
    satname: string
        short name for the satellite: 'L5', 'L7', 'L8' or 'S2'/'S2_RGB'
    cloud_mask_issue: boolean
        Set to True if there is an issue with the cloud mask and pixels are being
        erroneously masked on the images. It will check pixels around to determine
        if the cloud information is a mistake and shouldn't be taken into account.

    Returns:
    -----------
    cloud_mask : np.array
        boolean array with True if a pixel is cloudy and False otherwise  
    """

    # convert QA bits (the value of bits allocated to cloud cover vary depending on the satellite mission)
    if satname == 'L8':
        cloud_values = [2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908]
    elif satname == 'L7' or satname == 'L5':
        cloud_values = [752, 756, 760, 764]
    elif satname == 'S2' or satname == 'S2_RGB':
        cloud_values = [1024, 2048] # 1024 = dense clouds, 2048 = cirrus clouds

    # find which pixels have bits corresponding to cloud values
    cloud_mask = np.isin(im_QA, cloud_values)

    # remove cloud pixels that form very thin features. These can be for example beach or swash pixels
    # that are erroneously identified as clouds by the CFMASK algorithm applied to the images by the USGS
    if sum(sum(cloud_mask)) > 0 and sum(sum(~cloud_mask)) > 0:
        morphology.remove_small_objects(cloud_mask, min_size=10, connectivity=1, in_place=True)

        if cloud_mask_issue:
            elem = morphology.square(3) # use a square of width 3 pixels
            cloud_mask = morphology.binary_opening(cloud_mask,elem) # perform image opening
            # remove objects with less than 25 connected pixels
            morphology.remove_small_objects(cloud_mask, min_size=25, connectivity=1, in_place=True)

    return cloud_mask


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale (1-channel) image such that its histogram 
    matches that of a target image (panchromatic band). Used in pansharpening.

    Arguments:
    -----------
    source: np.array
        Image to transform; the histogram is computed over the flattened
        array
    template: np.array
        Template image; can have different dimensions than source
        
    Returns:
    -----------
    matched: np.array
        The transformed output image    
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to get 
    # the empirical cumulative distribution functions for the source and template
    # images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image that
    # correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def pansharpen(im_ms, im_pan, cloud_mask):
    """
    Pansharpens a multispectral image, using the panchromatic band and a cloud mask.
    A PCA is applied to the image, then the 1st PC is replaced, after histogram 
    matching with the panchromatic band. Note that it is essential to match the
    histograms of the 1st PC and the panchromatic band before replacing and 
    inverting the PCA.

    Arguments:
    -----------
    im_ms: np.array
        Multispectral image to pansharpen (3D - bands separated by GDAL)
    im_pan: np.array
        Panchromatic band (2D)
    cloud_mask: np.array
        2D cloud mask with True for cloud pixels

    Returns:
    -----------
    im_ms_ps: np.ndarray
        Pansharpened multispectral image (3D)
    """

    # reshape image into vector and apply cloud mask
    vec = im_ms.reshape(im_ms.shape[0] * im_ms.shape[1], im_ms.shape[2])
    vec_mask = cloud_mask.reshape(im_ms.shape[0] * im_ms.shape[1])
    vec = vec[~vec_mask, :]
    # apply PCA to multispectral bands
    pca = decomposition.PCA()
    vec_pcs = pca.fit_transform(vec)

    # replace 1st PC with pan band (after matching histograms)
    vec_pan = im_pan.reshape(im_pan.shape[0] * im_pan.shape[1])
    vec_pan = vec_pan[~vec_mask]
    vec_pcs[:,0] = hist_match(vec_pan, vec_pcs[:,0])
    vec_ms_ps = pca.inverse_transform(vec_pcs)

    # reshape vector into image
    vec_ms_ps_full = np.ones((len(vec_mask), im_ms.shape[2])) * np.nan
    vec_ms_ps_full[~vec_mask,:] = vec_ms_ps
    im_ms_ps = vec_ms_ps_full.reshape(im_ms.shape[0], im_ms.shape[1], im_ms.shape[2])

    return im_ms_ps


def rescale_image_intensity(im, cloud_mask, prob_high):
    """
    Rescales the intensity of an image (multispectral or single band) by applying
    a cloud mask and clipping the prob_high upper percentile. This functions allows
    to stretch the contrast of an image, only for visualisation purposes.

    Arguments:
    -----------
    im: np.array
        Image to rescale, can be 3D (multispectral) or 2D (single band)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    prob_high: float
        probability of exceedence used to calculate the upper percentile

    Returns:
    -----------
    im_adj: np.array
        rescaled image
    """

    # lower percentile is set to 0
    prc_low = 0

    # reshape the 2D cloud mask into a 1D vector
    vec_mask = cloud_mask.reshape(im.shape[0] * im.shape[1])

    # if image contains several bands, stretch the contrast for each band
    if len(im.shape) > 2:
        # reshape into a vector
        vec =  im.reshape(im.shape[0] * im.shape[1], im.shape[2])
        # initiliase with NaN values
        vec_adj = np.ones((len(vec_mask), im.shape[2])) * np.nan
        # loop through the bands
        for i in range(im.shape[2]):
            # find the higher percentile (based on prob)
            prc_high = np.percentile(vec[~vec_mask, i], prob_high)
            # clip the image around the 2 percentiles and rescale the contrast
            vec_rescaled = exposure.rescale_intensity(vec[~vec_mask, i],
                                                      in_range=(prc_low, prc_high))
            vec_adj[~vec_mask,i] = vec_rescaled
        # reshape into image
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1], im.shape[2])

    # if image only has 1 band (grayscale image)
    else:
        vec =  im.reshape(im.shape[0] * im.shape[1])
        vec_adj = np.ones(len(vec_mask)) * np.nan
        prc_high = np.percentile(vec[~vec_mask], prob_high)
        vec_rescaled = exposure.rescale_intensity(vec[~vec_mask], in_range=(prc_low, prc_high))
        vec_adj[~vec_mask] = vec_rescaled
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1])

    return im_adj


def preprocess_single(fn, satname, cloud_mask_issue):
    """
    Reads the image and outputs the pansharpened/down-sampled multispectral bands,
    the georeferencing vector of the image (coordinates of the upper left pixel),
    the cloud mask, the QA band and a no_data image. 
    For Landsat 7-8 it also outputs the panchromatic band, and for Sentinel 2 it
    also outputs the 20m SWIR band.

    Note that the methods implemented here are meant to be a quick hack, without 
    histogram correction techniques or additional steps that would be required 
    for a perfectly clean output.

    Arguments:
    -----------
    fn: str or list of str
        filename of the image .tif file. For L7, L8 and S2 this argument
        is a list of filenames, one filename for each resolution (30m 
        and 15m for Landsat 7-8, 10m, 20m and 60m for Sentinel 2)
    satname: str
        name of the satellite mission (e.g., 'L5')
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and wrong pixels are being masked

    Returns:
    -----------
    im_ms: np.array
        3D array (stored as a 1-channel image) containing the multispectral bands (B, G, R, NIR, SWIR1)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale] defining the coordinates
        of the top-left pixel of the image
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_extra : np.array
        2D array containing the 20m resolution SWIR band for Sentinel-2 and the 15m resolution
        panchromatic band for Landsat 7 and Landsat 8. This field is empty for Landsat 5.
    im_QA: np.array
        2D array containing the QA band, from which the cloud_mask can be computed.
    im_nodata: np.array
        2D array with True where no data values (-inf) are located.
    """

    #=============================================================================================#
    # L5 images
    #=============================================================================================#
    if satname == 'L5':

        # read all bands
        data = gdal.Open(fn, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)

        # down-sample to 15 m (half of the original pixel size)
        nrows = im_ms.shape[0]*2
        ncols = im_ms.shape[1]*2

        # create cloud mask
        im_QA = im_ms[:,:,5]
        im_ms = im_ms[:,:,:-1]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

        # resize the image using bilinear interpolation (order 1)
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')

        # adjust georeferencing vector to the new image size
        # scale becomes 15m and the origin is adjusted to the center of new top left pixel
        georef[1] = 15
        georef[5] = -15
        georef[0] = georef[0] + 7.5
        georef[3] = georef[3] - 7.5
        
        # check if -inf or nan values on any band and add to cloud mask
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # update cloud mask and nodata
        cloud_mask = np.logical_or(im_zeros, cloud_mask)
        im_nodata = np.logical_or(im_zeros, im_nodata)
        # no extra image for Landsat 5 (they are all 30 m bands)
        im_extra = []

    #=============================================================================================#
    # L7 images
    #=============================================================================================#
    elif satname == 'L7':

        # read pan image
        fn_pan = fn[0]
        data = gdal.Open(fn_pan, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_pan = np.stack(bands, 2)[:,:,0]

        # size of pan image
        nrows = im_pan.shape[0]
        ncols = im_pan.shape[1]

        # read ms image
        fn_ms = fn[1]
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)

        # create cloud mask
        im_QA = im_ms[:,:,5]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

        # resize the image using bilinear interpolation (order 1)
        im_ms = im_ms[:,:,:5]
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # update cloud mask and nodata
        cloud_mask = np.logical_or(im_zeros, cloud_mask)
        im_nodata = np.logical_or(im_zeros, im_nodata)

        # pansharpen Green, Red, NIR (where there is overlapping with pan band in L7)
        try:
            im_ms_ps = pansharpen(im_ms[:,:,[1,2,3]], im_pan, cloud_mask)
        except: # if pansharpening fails, keep downsampled bands (for long runs)
            im_ms_ps = im_ms[:,:,[1,2,3]]
        # add downsampled Blue and SWIR1 bands
        im_ms_ps = np.append(im_ms[:,:,[0]], im_ms_ps, axis=2)
        im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[4]], axis=2)

        im_ms = im_ms_ps.copy()
        # the extra image is the 15m panchromatic band
        im_extra = im_pan

    #=============================================================================================#
    # L8 images
    #=============================================================================================#
    elif satname == 'L8':

        # read pan image
        fn_pan = fn[0]
        data = gdal.Open(fn_pan, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_pan = np.stack(bands, 2)[:,:,0]

        # size of pan image
        nrows = im_pan.shape[0]
        ncols = im_pan.shape[1]

        # read ms image
        fn_ms = fn[1]
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_ms = np.stack(bands, 2)

        # create cloud mask
        im_QA = im_ms[:,:,5]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

        # resize the image using bilinear interpolation (order 1)
        im_ms = im_ms[:,:,:5]
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # update cloud mask and nodata
        cloud_mask = np.logical_or(im_zeros, cloud_mask)
        im_nodata = np.logical_or(im_zeros, im_nodata)

        # pansharpen Blue, Green, Red (where there is overlapping with pan band in L8)
        try:
            im_ms_ps = pansharpen(im_ms[:,:,[0,1,2]], im_pan, cloud_mask)
        except: # if pansharpening fails, keep downsampled bands (for long runs)
            im_ms_ps = im_ms[:,:,[0,1,2]]
        # add downsampled NIR and SWIR1 bands
        im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[3,4]], axis=2)

        im_ms = im_ms_ps.copy()
        # the extra image is the 15m panchromatic band
        im_extra = im_pan

    #=============================================================================================#
    # S2 images
    #=============================================================================================#

    # TODO: Add RE2 band processing (need to first add RE2 band download in sat_download).

    if satname == 'S2':

        # read 10m bands (B,G,R,NIR)
        fn10 = fn[0]
        data = gdal.Open(fn10, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im10 = np.stack(bands, 2)
        im10 = im10/10000 # TOA scaled to 10000

        # if image contains only zeros (can happen with S2), skip the image
        if sum(sum(sum(im10))) < 1:
            im_ms = []
            georef = []
            # skip the image by giving it a full cloud_mask
            cloud_mask = np.ones((im10.shape[0],im10.shape[1])).astype('bool')
            return im_ms, georef, cloud_mask, [], [], []

        # size of 10m bands
        nrows = im10.shape[0]
        ncols = im10.shape[1]

        # read 20m band (SWIR1)
        fn20 = fn[1]
        data = gdal.Open(fn20, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im20 = np.stack(bands, 2)
        im20 = im20[:,:,0]
        im20 = im20/10000 # TOA scaled to 10000

        # resize the image using bilinear interpolation (order 1)
        im_swir = transform.resize(im20, (nrows, ncols), order=1, preserve_range=True,
                                   mode='constant')
        im_swir = np.expand_dims(im_swir, axis=2)

        # append down-sampled SWIR1 band to the other 10m bands
        im_ms = np.append(im10, im_swir, axis=2)

        # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
        fn60 = fn[2]
        data = gdal.Open(fn60, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im60 = np.stack(bands, 2)
        im_QA = im60[:,:,0]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)
        # resize the cloud mask using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask,(nrows, ncols), order=0, preserve_range=True,
                                      mode='constant')
        # check if -inf or nan values on any band and add to cloud mask
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            cloud_mask = np.logical_or(np.logical_or(cloud_mask, im_inf), im_nan)
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)

        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # update cloud mask and nodata
        cloud_mask = np.logical_or(im_zeros, cloud_mask)
        im_nodata = np.logical_or(im_zeros, im_nodata)

        # the extra image is the 20m SWIR band
        im_extra = im20

    return im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata


def create_file(im_ms, cloud_mask, date, satname, filepath_out,
                georef, crs, formula, colormap, show):
    """
    Function meant to be called by save_file to create and save a new raster file (TIF format)
    corresponding to a user-selected band combination (either a custom formula, or a pre-
    defined index such as RGB or NDVI). For 1-band formulas, it also outputs a JPG
    visualization in a chosen colormap.

    Arguments:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B, G, R, NIR, SWIR1)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    date: str
        string containing the date at which the image was acquired
    satname: str
        name of the satellite mission (e.g., 'L5')
    filepath_out: str
        path to folder to save outputs (folder potentially created by save_file function)
    georef: list, optional
        georef vector to append to TIF file if image_format='tif'
    crs: int, optional
        crs to append to TIF file if image_foramt='tif'
    formula: str, optional (default: '[ RGB ]'')
        Remote sensing index or band combination to be computed using the preprocessed raster
        bands. Available indices are: RGB, NDVI, NDWI, BSI, AVI, NDSI, NDMI, SI, and valid 
        combinations are of the form: [ R * 0.5 , NIR * 1 , B * 0.7 ]. Spaces are important
        for the parser to understand the formula, and coefficients correspond to pixel intensity
        (in %). See more details here: sentinel-hub.com/sites/default/Custom_script_tutorial.pdf.
    colormap: str, optional (default: 'summer')
        Plot the index using a specific matplotlib colormap so that features
        are more visual. Possible colormaps are listed on: 
        https://matplotlib.org/examples/color/colormaps_reference.html
    show: bool, optional (default: False)
        output a visualization of the images as they are processed

    Returns:
    -----------
        Saves an image corresponding to the preprocessed satellite image.
    """

    # Rescale band intensity
    im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    im_NIR = rescale_image_intensity(im_ms[:,:,3], cloud_mask, 99.9)
    im_SWIR = rescale_image_intensity(im_ms[:,:,4], cloud_mask, 99.9)

    # Formula correspondence dictionary
    corres = {'R': 'im_RGB[...,0]',
        'G': 'im_RGB[...,1]',
        'B': 'im_RGB[...,2]',
        'RGB': 'im_RGB',
        'NIR': 'im_NIR',
        'SWIR': 'im_SWIR',
        'NDVI': '(im_NIR - im_RGB[...,0]) / (im_NIR + im_RGB[...,0])',
        'AVI': '(im_NIR * (1 - im_RGB[...,0]) * (im_NIR - im_RGB[...,0]))**(1/3)',
        'SI': '((1 - im_RGB[...,0]) * (1 - im_RGB[...,1]) * (1 - im_RGB[...,2]))**(1/3)',
        'BSI': '((im_SWIR + im_RGB[...,0]) - (im_NIR + im_RGB[...,2])) / ((im_SWIR + im_RGB[...,0]) + (im_NIR + im_RGB[...,2]))',
        'NDWI': '(im_NIR - im_RGB[...,0]) / (im_NIR + im_RGB[...,0])',
        'NDMI': '(im_NIR - im_SWIR) / (im_NIR + im_SWIR)',
        'NDSI': '(im_RGB[...,1] - im_SWIR) / (im_RGB[...,1] + im_SWIR)'
        } # you can just add any index here, as long as it uses R, G, B, NIR or SWIR (SWIR2) bands

    # Interpret the formula thanks to correspondence dictionary
    formula_stripped = formula.replace('[','').replace(']','').replace(' ','') # store stripped formula
    count = 0
    for key, value in corres.items():
        formula = formula.replace(' '+key+' ', value) 
        count += 1
    assert count > 0, 'Please enter a valid formula. Valid formulas include band combinations' + \
                      ' written such as [ R * 0.5 , NIR * 1 , B * 0.7 ] (with spaces everywhere)' + \
                      ' and predefined formulas among: {}'.format(list(corres.keys()))
    
    # Execute the formula with eval function
    im_formula = np.asarray(eval(formula)) # thanks to the assertion above, eval content is safe
    #print('1 ', im_formula.shape)
    if len(im_formula.shape) > 3:
        im_formula = np.squeeze(im_formula)
    #print('2 ',im_formula.shape)
    if im_formula.shape[2] > 3:
        im_formula = np.moveaxis(im_formula, 0, -1) # put channels as last axis to reshape as image
    #    print('3 ',im_formula.shape)
    im_formula = skimage.img_as_ubyte(im_formula) # convert to uint8 (TODO: check if better before applying formula)
    #print('4 ',im_formula.shape)
    im_formula = im_formula.astype('int')

    # Write the new raster with gdal
    dst_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(filepath_out, date + '_' + satname +\
                                      '_' + formula_stripped + '.tif'),
                                      im_formula.shape[1], im_formula.shape[0], im_formula.shape[2], 
                                      gdal.GDT_Byte)
    dst_ds.SetGeoTransform(georef) # specify coords
    srs = osr.SpatialReference() # establish encoding
    srs.ImportFromEPSG(crs) # set to original image crs
    dst_ds.SetProjection(srs.ExportToWkt()) # export crs to tiff-understandable format
    for i in range(im_formula.shape[2]):
        # print(im_formula.shape)
        dst_ds.GetRasterBand(i+1).WriteArray(im_formula[...,i]) # write all bands to raster
    dst_ds.FlushCache() # write to disk
    dst_ds = None # close process

    # If final raster has only 1 band, also save as png with specified colormap for visualization
    if im_formula.shape[2] == 1:
        im_formula = np.squeeze(im_formula)
        plt.imsave(os.path.join(filepath_out, date + '_' + satname + '_' + formula_stripped + '.png'),
                   im_formula, cmap=plt.get_cmap(colormap))

    # show example
    if show:
        if im_formula.shape[2] == 1:
            fig = plt.figure(figsize = (10,10))
            plt.title('Image preview')
            plt.imshow(im_formula, cmap=plt.get_cmap(colormap))
        else:
            fig = plt.figure(figsize = (10,10))
            plt.title('Image preview')
            plt.imshow(im_formula)

    # If needed, plot RGB image as a figure with title, custom size and resolution, etc.
    # fig = plt.figure()
    # fig.set_size_inches([18,9])
    # ax1 = fig.add_subplot(111)
    # ax1.axis('off')
    # ax1.imshow(im_RGB)
    # ax1.set_title(date + '   ' + satname, fontsize=16)

    # Same, but with other bands in the plot alongside RGB
   # if im_RGB.shape[1] > 2*im_RGB.shape[0]:
   #     ax1 = fig.add_subplot(311)
   #     ax2 = fig.add_subplot(312)
   #     ax3 = fig.add_subplot(313)
   # else:
   #     ax1 = fig.add_subplot(131)
   #     ax2 = fig.add_subplot(132)
   #     ax3 = fig.add_subplot(133)
   # # RGB
   # ax1.axis('off')
   # ax1.imshow(im_RGB)
   # ax1.set_title(date + '   ' + satname, fontsize=16)
   # # NIR
   # ax2.axis('off')
   # ax2.imshow(im_NIR, cmap='seismic')
   # ax2.set_title('Near Infrared', fontsize=16)
   # # SWIR
   # ax3.axis('off')
   # ax3.imshow(im_SWIR, cmap='seismic')
   # ax3.set_title('Short-wave Infrared', fontsize=16)

    # Save figure
    # plt.rcParams['savefig.jpeg_quality'] = 100 # disable jpeg lossy image compression
    # fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi = 150) 
    
    # Alternative saving parameters (no margins, higher resolution)
    # fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), 
    #      bbox_inches = 'tight', pad_inches = 0, dpi=1000)
    # plt.close()


def save_files(metadata, inputs, formula='[ RGB ]', colormap='summer',
               show=False, cloud_mask_issue=False):
    """
    Creates and saves a new raster file (TIF format) corresponding to a user-selected 
    band combination (either a custom formula, or a pre-defined index such as RGB or 
    NDVI), for all the images contained in metadata. For 1-band formulas, it also 
    saves a JPG visualization in a chosen colormap.

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    inputs: dict 
        input parameters from sat_download (sitename, filepath, polygon, dates, sat_list)    
    cloud_mask_issue: boolean, optional (default: False)
            True if there is an issue with the cloud mask and pixels
            are erroneously being masked on the images
    formula: str, optional (default: '[ RGB ]')
        Remote sensing index or band combination to be computed using the preprocessed raster
        bands. Available indices are: RGB, NDVI, NDWI, BSI, AVI, NDSI, NDMI, SI, and valid 
        combinations are of the form: [ R * 0.5 , NIR * 1 , B * 0.7 ]. Spaces are important
        for the parser to understand the formula, and coefficients correspond to pixel intensity
        (in %). See more details here: sentinel-hub.com/sites/default/Custom_script_tutorial.pdf.
    colormap: str, optional (default: 'summer')
        Plot the index using a specific matplotlib colormap so that features
        are more visual. Possible colormaps are listed on: 
        https://matplotlib.org/examples/color/colormaps_reference.html
    show: bool, optional (default: False)
        output a visualization of the images as they are processed
    
    Returns:
    -----------
        Stores the images in a folder named /preprocessed.
    """

    sitename = inputs['sitename']
    filepath_data = inputs['filepath']
    formula_stripped = formula.replace('[','').replace(']','').replace(' ', '')

    print('Starting preprocessing as {}...'.format(formula_stripped))

    # create subfolder to store the files
    if len(formula) <= 8:
        filepath_out = os.path.join(filepath_data, sitename, 'preprocessed_files',
                                   '{}'.format(formula_stripped))
    else:
        filepath_out = os.path.join(filepath_data, sitename, 'preprocessed_files', 'custom_formula')
    if not os.path.exists(filepath_out):
            os.makedirs(filepath_out)

    # loop through satellite list
    for satname in metadata.keys():

        filepath = sat_tools.get_filepath(inputs,satname)
        filenames = metadata[satname]['filenames']

        # loop through images
        for i in range(len(filenames)):
            # image filename
            fn = sat_tools.get_filenames(filenames[i],filepath, satname)
            # get crs
            crs = int(metadata[satname]['epsg'][i])
            # read and preprocess image
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess_single(fn, satname, cloud_mask_issue)
            # create and save file with requested parameters
            date = filenames[i][:-4]
            plt.ioff()  # turning interactive plotting off (in case we use the plotting options in create_file)
            try:
                create_file(im_ms, cloud_mask, date, satname, filepath_out,
                            georef, crs, formula, colormap, show)
            except:
                continue
            # print percentage completion for user
            print('\r%d%%' %int((i+1)/len(filenames)*100), end='')

    # print the location where the images have been saved
    print('\n{} satellite images saved in {}'.format(formula_stripped, filepath_out))


def timelapse(timelapse_settings, correct_path_row=False):
    """
    Saves a timelapse video (.mp4) of images corresponding to the arguments
    in the Google Drive linked to the user's GEE account.

    Available images are Landsat 5 (03/1984 - 05/2013), Landsat 7 (04/1999 - present,
    but several system failures make Landsat 8 a preferred choice when possible),
    and Landsat 8 (06/2013 - present).

    TODO: 1) download video directly on disk, 2) add Sentinel-2 and MODIS, 3) add 
    remote sensing indices timelapse, 4) merge sources if dates spans over several
    satellite periods.

    Arguments:
    -----------
    timelapse_settings: dict, with possible keys:
        sat: str
            'L5', 'L7' or 'L8' (careful with satellite working periods)
        dates: list of str
            start and end dates in format: yyyy-mm-dd
        max_cloud_cover: int
            maximum cloud cover percentage
        bands: list
            bands to select, in format ['R', 'G', 'B'], among 'R', 'G', 'B', 'NIR',
            'SWIR1', 'SWIR2', 'PAN' (and 'CA' for Landsat 8)
        filename: str
            name of the video to be saved on Drive
        bbox: list
            list of (lon, lat) tuples corresponding to summits of the area of interest -
            note that you need to pass either a bbox or a point and radius
        point: tuple
            center of the area of interest in (lon, lat) format (ex: (-122.7286, 37.6325))
        radius: int
            radius of the area of interest, in km
        video_dim: int, optional (default: 720)
            width of the output video - choose a higer value to increase quality
        frames_per_sec: int, optional (default: 12)
            nb of images per second in the output video - choose a higher value for
            a smoother result
    correct_path_row: boolean, optional (default: False)
        set to True if the output video shows black areas, meaning that your area
        overlaps on 2 satellite areas - the function will run a new search to 
        correct the automatic Landsat image selection

    Returns:
    -----------
    saves timelapse as {filename}.mp4 in the user's Google Drive account linked to GEE
    """

    # Initializations
    ee.Initialize()
    sat_corres = {'L5':'LANDSAT/LT05/C01/T1_TOA', 
              'L7':'LANDSAT/LE07/C01/T1_TOA',
              'L8':'LANDSAT/LC08/C01/T1_TOA'}
    band_corres_L5_L7 = {'R':'B3', 'G':'B2', 'B':'B1', 'NIR':'B4',
                         'SWIR1':'B5', 'TIR':'B6', 'SWIR2':'B7', 'PAN':'B8'}
    band_corres_L8 = {'R':'B4', 'G':'B3', 'B':'B2', 'NIR':'B5',
                      'SWIR1':'B6', 'SWIR2':'B7', 'PAN':'B8', 'CA':'B1'}
    if 'video_dim' not in timelapse_settings:
        timelapse_settings['video_dim'] = 720
    if 'frames_per_sec' not in timelapse_settings:
        timelapse_settings['frames_per_sec'] = 12
    bands = timelapse_settings['bands']

    # Choose image collection
    sat = timelapse_settings['sat']
    collection = ee.ImageCollection(sat_corres[sat])

    # Define area
    assert ('point' in timelapse_settings) or ('bbox' in timelapse_settings),\
                                             'Please pass point & radius or bbox coordinates.'
    if 'point' in timelapse_settings:
        assert 'radius' in timelapse_settings, "Don't forget to pass the radius of the area of study."
        center = ee.Geometry.Point(timelapse_settings['point'])
        user_bbox = sat_tools.bbox_from_point(timelapse_settings['point'], timelapse_settings['radius'])
        user_bbox = ee.Geometry.Polygon(user_bbox)
    elif 'bbox' in timelapse_settings:
        user_bbox = ee.Geometry.Polygon(timelapse_settings['bbox'])
        center = shapely.geometry.Polygon(timelapse_settings['bbox']).centroid.coords[0]
        center = ee.Geometry.Point(center)
    collection = collection.filterBounds(center)

    # Correct Landsat path and row selection if needed
    if correct_path_row:
        print('Correcting Landsat tile selection...')
        path, row = sat_tools.find_path_row(point)
        collection = collection.filter(ee.Filter.eq('WRS_PATH', path))
        collection = collection.filter(ee.Filter.eq('WRS_ROW', row))
        print('Done, now applying date and cloud filters.')
 
    # Apply date and cloud filters
    collection = collection.filter(ee.Filter.lt('CLOUD_COVER', timelapse_settings['max_cloud_cover']))
    collection = collection.filterDate(timelapse_settings['dates'][0], timelapse_settings['dates'][1])

    # Select the bands and convert to 8-bit
    if timelapse_settings['sat'] == 'L8':
        user_bands = [band_corres_L8[bands[0]], band_corres_L8[bands[1]], band_corres_L8[bands[2]]]
    else:
        user_bands = [band_corres_L5_L7[bands[0]], band_corres_L5_L7[bands[1]], band_corres_L5_L7[bands[2]]]
    collection = collection.select(user_bands)
    def convertBit(image):
        return image.multiply(512).uint8()
    collection = collection.map(convertBit)
    
    # Print info for user
    count = collection.size()
    #print("{} images successfully selected.".format(count))
    print("Processing timelapse on GEE server...")

    # Export to video and launch processing on GEE
    video = batch.Export.video.toDrive(collection, description=timelapse_settings['filename'],
                                 dimensions = timelapse_settings['video_dim'],
                                 framesPerSecond = timelapse_settings['frames_per_sec'],
                                 region=user_bbox.getInfo()["coordinates"])
    __ = batch.Task.start(video)

    # Monitor the task
    while video.status()['state'] in ['READY', 'RUNNING']:
        print(video.status()['state'])
        time.sleep(10)
    else:
        print(video.status()['state'] + ' - Please check on your Google Drive account.')

