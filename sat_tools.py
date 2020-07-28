"""
This module contains utilities to work with satellite images, in
particular functions related to geographic computations, an image
annotator using the matplotlib console, and an image slicer sub-module.
    
Original author: Kilian Vos, Water Research Laboratory,
                 University of New South Wales, 2018
Modifications and additions: Thomas de Mareuil, Total E-LAB, 2020
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
from osgeo import gdal, osr
import geopandas as gpd
import pandas as pd
from shapely import geometry
import skimage.transform as transform
from astropy.convolution import convolve
import math
from pylab import ginput
import pickle
import folium
from folium.plugins import MarkerCluster
import json
import csv
import geopy
import pyproj
from PIL import Image
from natsort import realsorted
import glob

# other totalsat modules
from totalsat import sat_preprocess, sat_tools


###################################################################################################
# COORDINATES CONVERSION AND REVERSE GEOCODING FUNCTIONS
###################################################################################################

def convert_pix2world(points, georef):
    """
    Converts pixel coordinates (pixel row and column) to world projected 
    coordinates performing an affine transformation.

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (row first and column second)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates, first columns with X and second column with Y
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            tmp = arr[:,[1,0]]
            points_converted.append(tform(tmp))
          
    # if single array
    elif type(points) is np.ndarray:
        tmp = points[:,[1,0]]
        points_converted = tform(tmp)
        
    else:
        raise Exception('invalid input type')
        
    return points_converted


def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates 
    (pixel row and column) performing an affine transformation.

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (X,Y)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates (pixel row and column)
    
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)
    
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(tform.inverse(points))
            
    # if single array    
    elif type(points) is np.ndarray:
        points_converted = tform.inverse(points)
        
    else:
        print('invalid input type')
        raise
        
    return points_converted


def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes.

    Arguments:
    -----------
    points: np.array or list of np.ndarray
        array with 2 columns (rows first and columns second)
    epsg_in: int
        epsg code of the spatial reference in which the input is
    epsg_out: int
        epsg code of the spatial reference in which the output will be            
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates from epsg_in to epsg_out
        
    """
    
    # define input and output spatial references
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    # create a coordinates transform
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))
    # if single array
    elif type(points) is np.ndarray:
        points_converted = np.array(coordTransform.TransformPoint(points))
        # note: I removed the 's' in 'TransformPoints' - might need to add it back if bug
    else:
        raise Exception('invalid input type')

    return points_converted


def bbox_from_point(point, radius):
    """
    Computes a square bounding box with chosen radius around point coordinates.

    Arguments:
    -----------
    point: tuple
        point coordinates in format: (lon,lat)
    radius: int
        radius (in km) of bounding box around center coordinates
    
    Returns:
    -----------
    bbox: list
        list of coordinates (lon, lat) of the 4 bounding box summits
    """

    # We use the approximations: 1 decimal degree of latitude = 111.321 km
    #                            1 decimal degree of longitude = cos(latitude in DD * pi/180) * 111.321 km
    # where 111.321 km is the length of a DD of longitude at the equator and pi/180 converts DDs to radians.
    x_min = round(point[0], 3) - \
            round(radius / (math.cos(round(point[1],3) * math.pi/180) * 111.321), 3)
    x_max = round(point[0], 3) + \
            round(radius / (math.cos(round(point[1],3) * math.pi/180) * 111.321), 3)
    y_min = round(point[1], 3) - round(radius/111.321, 3)
    y_max = round(point[1], 3) + round(radius/111.321, 3)
    
    return [[x_min, y_max],[x_max, y_max],[x_max, y_min],[x_min, y_min]]


def split_area(bbox, pixel_width, max_size):
    """
    Splits a square polygon in smaller squares.

    Arguments:
    -----------
    bbox: list
        list of coordinates (lon, lat) of the 4 bounding box summits - must be in
        the order: [[x_min, y_max],[x_max, y_max],[x_max, y_min],[x_min, y_min]]
    pixel_width: int
        width of the bbox, in pixels
    max_size: int
        maximum authorized pixel width

    Returns:
    -----------
    small_boxes: list
        list of smaller boxes (with coordinates in the same format as the original)
    """    
    
    # get coordinates of the original bbox     
    x_min = bbox[0][0]
    y_max = bbox[0][1]
    x_max = bbox[2][0]
    y_min = bbox[2][1]

    # compute size of each small box
    small_boxes = []
    nb_breaks = int(pixel_width // max_size)
    x_step = (x_max - x_min) / (nb_breaks+1)
    y_step = (y_max - y_min) / (nb_breaks+1)

    # compute coordinates of each small box
    for i in range(nb_breaks+1):
        x_min_i = x_min + i*x_step
        x_max_i = x_max - (nb_breaks-i)*x_step
        for j in range(nb_breaks+1):
            y_min_i = y_min + j*y_step
            y_max_i = y_max - (nb_breaks-j)*y_step
            box_i = [[x_min_i, y_max_i],[x_max_i, y_max_i],[x_max_i, y_min_i],[x_min_i, y_min_i]]
            small_boxes.append(box_i)
    
    return small_boxes


def find_path_row(point):
    """
    Finds the Landsat WRS2 path and row number that best corresponds to any (lat,lon) point.
    Differs from other implementations where the list of overlapping rows is returned. Here,
    we return only the path/row combination whose center is closest to the point.

    Arguments:
    -----------
    point: tuple
        point coordinates in format: (lon, lat)

    Returns:
    -----------
    path, row: int, int
        closest WRS2 path/row combination
    """

    # open the path and row list from online resource
    url = "https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
    open_url = urllib.request.urlopen(url)
    
    # extract it in temporary file
    with zipfile.ZipFile(io.BytesIO(open_url.read())) as zip_file:
        tempdir = tempfile.mkdtemp()
        zip_file.extractall(tempdir)
        # data = ogr.Open(os.path.join(tempdir, 'WRS2_descending.shp'))
        # layer = data.GetLayer(0)
        table = gpd.read_file(os.path.join(tempdir, 'WRS2_descending.shp'))

    # clean up after ourselves
    for file in os.listdir(tempdir):
        os.unlink(os.path.join(tempdir, file))
    os.rmdir(tempdir)
    
    # compute center of each path/row
    table['centroid'] = table['geometry'].centroid
    table = table[['PATH','ROW','centroid']]
    table['centroid'] = table['centroid'].apply(lambda x: x.coords[0][::-1])
    
    # get the closest path/row from point
    closest = min(list(table['centroid']), key=lambda x: geopy.distance.distance(point[::-1], x).km)
    path = int(table[table['centroid'] == closest]['PATH'])
    row = int(table[table['centroid'] == closest]['ROW'])
    
    return path, row


def get_image_centroid(image_path):
    """
    Get (lat, lon) coordinates of the center of a georeferenced image (.tif), in epsg:4326.

    Arguments:
    -----------
    image_path: str
        path to TIF image file

    Returns:    
    -----------
    centroid: tuple
        (lat, lon) centroid coordinates, in epsg:4326, with 6 decimals
    """
    
    # read image
    try:
        ds = gdal.Open(image_path)
    except:
        raise Exception('Please provide the path to a georeferenced image.')
    
    # compute centroid coordinates
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5] 
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]
    centroid = (np.mean([minx,maxx]), np.mean([miny,maxy]))

    # reproject to epsg:4326
    epsg_in = int(osr.SpatialReference(wkt=ds.GetProjection()).GetAttrValue('AUTHORITY',1))
    inProj = pyproj.Proj('epsg:'+str(epsg_in))
    outProj = pyproj.Proj('epsg:4326')
    lat, lon = pyproj.transform(inProj, outProj, centroid[0], centroid[1])
    
    return (round(lat,6), round(lon,6))


def get_address(coordinates, parsing='full_address_df'):
    """
    Get address (city, postcode, etc.) from geographic coordinates, using the 
    geopy package.

    Arguments:
    -----------
    coordinates: tuple
        (lat, lon) coordinates
    parsing: str or list of str, optional (default: 'full_address_df')
        chosen part of the adress to output - by default, the function will output the 
        whole address as a dataframe. Other possibilities are 'full_address_str' and 
        one or several keys among: ['country', 'state', 'county', 'postcode', city',  
        'road', 'house_number']. Available keys depend on OSM information for the 
        specified location, and if the chosen keys aren't available the function will 
        fill the blank with a NaN.

    Returns:    
    -----------
    address: str or pd.DataFrame
        result of the parsing specified by user (by default, full address as a dataframe)
    """
    
    # Nominatim keys, for reference
    # all_keys = ['country', 'state', 'county', 'municipality', 'postcode', 'town', 'village', 'city', 
    #             'city_district', 'hamlet', 'isolated_dwelling', 'croft', 'quarter', 'district', 'borough',
    #             'neighbourhood', 'allotments', 'subdivision', 'suburb', 'road', 'house_number', 'house_name']

    # query address
    locator = geopy.geocoders.Nominatim(user_agent="me", timeout=10)
    location = locator.reverse(coordinates, zoom=18) # zoom determines the precision of the address

    # default parsing
    if parsing == 'full_address_df':
        parsed_address = location.raw['address']
        df_address = pd.DataFrame(parsed_address, columns=list(parsed_address.keys()), index=[0])
        df_address['lat'] = coordinates[0]
        df_address['lon'] = coordinates[1]
        return df_address

    # user-defined parsing
    elif parsing == 'full_address_str':
        full_address_str = location.address
        # postcode = [x.strip() for x in full_address.split(',')][-2] # example code to parse from string
        return full_address_str
    
    else:
        parsed_address = location.raw['address']
        user_parsing = {}
        if type(parsing) == str:
            parsing = [parsing]
        for el in parsing:
            try:
                res = parsed_address[el]
            # Below, we deal with the case when the specified parsing key isn't in the raw address.
            # We go through the alternative keys for 'city' or 'house_number', and if it still yields
            # no result, we set the result to None. I guess there are more sythetic ways to do this - 
            # but it works well or our use case.
            except:
                if el == 'city':
                    alternative_el = 'town'
                    try:
                        res = parsed_address[alternative_el]
                    except:
                        alternative_el = 'village'
                        try:
                            res = parsed_address[alternative_el]
                        except:
                            res = None
                elif el == 'house_number':
                    alternative_el = 'house_name'
                    try:
                        res = parsed_address[alternative_el]
                    except:
                        res = None
                else:
                    res = None
            user_parsing[el] = res
        
        if len(user_parsing) == 1:
            return user_parsing[parsing[0]]
        else:
            df_address = pd.DataFrame(user_parsing, columns=list(user_parsing.keys()), index=[0])
            df_address['lat'] = coordinates[0]
            df_address['lon'] = coordinates[1]
            return df_address


###################################################################################################
# IMAGE ANALYSIS FUNCTIONS
###################################################################################################
    
def nd_index(im1, im2, cloud_mask):
    """
    Computes normalised difference index on 2 images (2D), given a cloud mask (2D).

    Arguments:
    -----------
    im1: np.array
        first image (2D) with which to calculate the ND index
    im2: np.array
        second image (2D) with which to calculate the ND index
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:    
    -----------
    im_nd: np.array
        Image (2D) containing the ND index
    """

    # reshape the cloud mask
    vec_mask = cloud_mask.reshape(im1.shape[0] * im1.shape[1])
    # initialise with NaNs
    vec_nd = np.ones(len(vec_mask)) * np.nan
    # reshape the two images
    vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
    vec2 = im2.reshape(im2.shape[0] * im2.shape[1])
    # compute the normalised difference index
    temp = np.divide(vec1[~vec_mask] - vec2[~vec_mask],
                     vec1[~vec_mask] + vec2[~vec_mask])
    vec_nd[~vec_mask] = temp
    # reshape into image
    im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])

    return im_nd

    
def image_std(image, radius):
    """
    Calculates the standard deviation of an image, using a moving window of 
    specified radius. Uses astropy's convolution library.
    
    Arguments:
    -----------
    image: np.array
        2D array containing the pixel intensities of a single-band image
    radius: int
        radius defining the moving window used to calculate the standard deviation. 
        For example, radius = 1 will produce a 3x3 moving window.
        
    Returns:    
    -----------
    win_std: np.array
        2D array containing the standard deviation of the image
        
    """  
    
    # convert to float
    image = image.astype(float)
    # first pad the image
    image_padded = np.pad(image, radius, 'reflect')
    # window size
    win_rows, win_cols = radius*2 + 1, radius*2 + 1
    # calculate std with uniform filters
    win_mean = convolve(image_padded, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_sqr_mean = convolve(image_padded**2, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_var = win_sqr_mean - win_mean**2
    win_std = np.sqrt(win_var)
    # remove padding
    win_std = win_std[radius:-radius, radius:-radius]

    return win_std


def mask_raster(fn, mask):
    """
    Masks a .tif raster using GDAL.
    
    Arguments:
    -----------
    fn: str
        filepath + filename of the .tif raster
    mask: np.array
        array of boolean where True indicates the pixels that are to be masked
        
    Returns:    
    -----------
    Overwrites the .tif file directly
        
    """ 
    
    # open raster
    raster = gdal.Open(fn, gdal.GA_Update)
    # mask raster
    for i in range(raster.RasterCount):
        out_band = raster.GetRasterBand(i+1)
        out_data = out_band.ReadAsArray()
        out_band.SetNoDataValue(0)
        no_data_value = out_band.GetNoDataValue()
        out_data[mask] = no_data_value
        out_band.WriteArray(out_data)
    # close dataset and flush cache
    raster = None


###################################################################################################
# UTILITIES
###################################################################################################
    
def get_filepath(inputs,satname):
    """
    Create filepath to the different folders containing the satellite images.

    Arguments:
    -----------
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in 
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include: 
            ```
            sat_list = ['L5', 'L7', 'L8', 'S2']
            ```
        'filepath_data': str
            filepath to the directory where the images are downloaded
    satname: str
        short name of the satellite mission ('L5','L7','L8','S2')
                
    Returns:    
    -----------
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images
    
    """     
    
    sitename = inputs['sitename']
    filepath_data = inputs['filepath']
    # access the images
    if satname == 'L5':
        # access downloaded Landsat 5 images
        filepath = os.path.join(filepath_data, sitename, satname, '30m')
    elif satname == 'L7':
        # access downloaded Landsat 7 images
        filepath_pan = os.path.join(filepath_data, sitename, 'L7', 'pan')
        filepath_ms = os.path.join(filepath_data, sitename, 'L7', 'ms')
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'L8':
        # access downloaded Landsat 8 images
        filepath_pan = os.path.join(filepath_data, sitename, 'L8', 'pan')
        filepath_ms = os.path.join(filepath_data, sitename, 'L8', 'ms')
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'S2':
        # access downloaded Sentinel 2 images
        filepath10 = os.path.join(filepath_data, sitename, satname, '10m')
        filepath20 = os.path.join(filepath_data, sitename, satname, '20m')
        filepath60 = os.path.join(filepath_data, sitename, satname, '60m')
        filepath = [filepath10, filepath20, filepath60]
            
    return filepath
  

def get_filenames(filename, filepath, satname):
    """
    Creates filepath + filename for all the bands belonging to the same image.

    Arguments:
    -----------
    filename: str
        name of the downloaded satellite image as found in the metadata
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images
    satname: str
        short name of the satellite mission       
        
    Returns:    
    -----------
    fn: str or list of str
        contains the filepath + filenames to access the satellite image
  
    """     
    
    if satname == 'L5':
        fn = os.path.join(filepath, filename)
    if satname == 'L7' or satname == 'L8':
        filename_ms = filename.replace('pan','ms')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename_ms)]
    if satname == 'S2':
        filename20 = filename.replace('10m','20m')
        filename60 = filename.replace('10m','60m')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename20),
              os.path.join(filepath[2], filename60)]
        
    return fn


def merge_output(output):
    """
    Function to merge the output dictionary, which has one key per satellite mission
    into a dictionnary containing all the shorelines and dates ordered chronologically.
    
    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding dates, organised by 
        satellite mission
    
    Returns:    
    -----------
    output_all: dict
        contains the extracted shorelines in a single list sorted by date
    
    """     
    
    # initialize output dict
    output_all = dict([])
    satnames = list(output.keys())
    for key in output[satnames[0]].keys():
        output_all[key] = []
    # create extra key for the satellite name
    output_all['satname'] = []
    # fill the output dict
    for satname in list(output.keys()):
        for key in output[satnames[0]].keys():
            output_all[key] = output_all[key] + output[satname][key]
        output_all['satname'] = output_all['satname'] + [_ for _ in np.tile(satname,
                  len(output[satname]['dates']))]
    # sort chronologically
    idx_sorted = sorted(range(len(output_all['dates'])), key=output_all['dates'].__getitem__)
    for key in output_all.keys():
        output_all[key] = [output_all[key][i] for i in idx_sorted]

    return output_all


def create_gif(input_folder, extension, output_path, duration=200, sort='numerical'):
    """
    Creates a gif from images stored in a specified folder.
    
    Arguments:
    -----------
    input_folder: str
        path to the folder with images to convert to gif
    extension: str
        extension of the images ('.tif', '.jpg', '.png', including the point)
    output_path: str
        path and full name of the gif to create (ex: Desktop/mycreation.gif)
    duration: int, optional (default: 200)
        duration of the gif loop, in ms/image
    sort: str, optional (default: 'numerical')
        logic for sorting filenames - either  'alphabetical' or 'numerical', 
        if image names include a numeric information (this method makes use of 
        the natsort library, https://github.com/SethMMorton/natsort, to sort 
        in real numbers order instead of alphanumerical in order to avoid
        incoherence such as '1,10,2,3...')

    Returns:    
    -----------
    Saves a gif file at the specified output path.
    """
    
    # load and sort images
    if len(extension) < 4: # add point to extension if it has been forgotten
        extension = '.' + extension
    filenames = glob.glob(os.path.join(input_folder, '*') + extension)
    if sort == 'numerical':
        filenames = realsorted(filenames)
    elif sort == 'alphabetical':
        filenames.sort()
    else:
        raise Exception("Available sorting logics are 'numerical' and 'alphabetical'.")

    # store them into a list of PIL images
    frames = []
    for i in filenames:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(output_path,
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=duration, loop=0)
    print('GIF saved at ' + output_path)


def plot_geometries(geometries, init_crs=4326):
    """
    Plots geometries in an interactive folium map, for visualization.

    Arguments:
    -----------
    geometries: list or geodataframe
        supports geodataframes with a 'geometry' column and list of points (lon,lat) or polygons
        (formatted as lists of lon,lat coordinates)
    init_crs: int, optional (default: 4326)
        geometries crs, if not WSG84

    Returns:
    -----------
        folium map with geometries as map overlay
    """

    # if the geometries are not a gdf (i.e. a polygon or list of polygons)
    if str(type(geometries)) != "<class 'geopandas.geodataframe.GeoDataFrame'>":
        
        # workflow if we pass a single geometry
        try:
            is_unique = geometries.shape # if it's not a list, it will throw an error and move below
            # if it's a list of point coordinates, convert to Polygon object
            if len(geometries) > 2:
                geometries_shp = geometry.Polygon(geometries)
                map_lat = geometries[0][1]
                map_lon = geometries[0][0]
            # if it's a single point coordinates, convert to Point object
            else:
                geometries_shp = geometry.Point(geometries)
                map_lat = geometries[1]
                map_lon = geometries[0]
            # convert to geodataframe and make sure we are in epsg:4326
            geometries_gdf = gpd.GeoDataFrame(index=[0], crs={'init':'epsg:'+str(init_crs)}, geometry=[geometries_shp])
            geometries_gdf = geometries_gdf.to_crs(epsg=4326)

        # workflow if we pass a list of geometries
        except:
            # if it's a list of polygons, convert each one to a Polygon object
            if len(geometries[0]) > 2:
                geometries_shp = [geometry.Polygon(poly) for poly in geometries]
                map_lat = geometries[0][0][1]
                map_lon = geometries[0][0][0]
            # if it's a list of points, convert each one to a Point object
            else:
                geometries_shp = [geometry.Point(poly) for poly in geometries]
                map_lat = geometries[0][1]
                map_lon = geometries[0][0]
            # convert to geodataframe and make sure we are in epsg:4326
            geometries_gdf = gpd.GeoDataFrame(crs={'init':'epsg:'+str(init_crs)}, geometry=geometries_shp)
            geometries_gdf = geometries_gdf.to_crs(epsg=4326)

    # if the geometries are stored in a geodataframe, just convert to epsg:4326
    else:
        geometries_gdf = geometries.to_crs(epsg=4326)
        map_lat = list(list(geometries_gdf['geometry'])[0].exterior.coords)[0][1]
        map_lon = list(list(geometries_gdf['geometry'])[0].exterior.coords)[0][0]

    # Plot geometries on an interactive map
    m = folium.Map([map_lat, map_lon], zoom_start=12)
    folium.GeoJson(geometries_gdf, name='geometries').add_to(m)

    # Add information popups on geometries
    locations = []
    popups = []
    for idx, row in geometries_gdf.iterrows():
        locations.append([row['geometry'].centroid.y, row['geometry'].centroid.x])
        popups.append("Coordinates: {}".format((round(row['geometry'].centroid.y,6), 
                                                round(row['geometry'].centroid.x,6))))
    s = folium.FeatureGroup(name='popup info')
    s.add_child(MarkerCluster(locations=locations, popups=popups))
    m.add_child(s)

    # Add satellite layer
    tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)

    # Add layer control and lat/lng popups
    folium.LatLngPopup().add_to(m)
    folium.LayerControl().add_to(m)
    
    display(m)


###################################################################################################
# CONVERSIONS BETWEEN FILE FORMATS (GEODATAFRAME, GEOJSON, etc.)
###################################################################################################
    
def polygon_from_kml(fn):
    """
    Extracts coordinates from a .kml file.

    Arguments:
    -----------
    fn: str
        filepath + filename of the kml file to be read          
                
    Returns:    
    -----------
    polygon: list
        coordinates extracted from the .kml file
        
    """    
    
    # read .kml file
    with open(fn) as kmlFile:
        doc = kmlFile.read() 
    # parse to find coordinates field
    str1 = '<coordinates>'
    str2 = '</coordinates>'
    subdoc = doc[doc.find(str1)+len(str1):doc.find(str2)]
    coordlist = subdoc.split('\n')
    # read coordinates
    polygon = []
    for i in range(1,len(coordlist)-1):
        polygon.append([float(coordlist[i].split(',')[0]), float(coordlist[i].split(',')[1])])
        
    return [polygon]

def transects_from_geojson(filename):
    """
    Reads transect coordinates from a .geojson file.
    
    Arguments:
    -----------
    filename: str
        contains the path and filename of the geojson file to be loaded
        
    Returns:    
    -----------
    transects: dict
        contains the X and Y coordinates of each transect
        
    """  
    
    gdf = gpd.read_file(filename)
    transects = dict([])
    for i in gdf.index:
        transects[gdf.loc[i,'name']] = np.array(gdf.loc[i,'geometry'].coords)
        
    print('%d transects have been loaded' % len(transects.keys()))

    return transects

def output_to_gdf(output):
    """
    Saves the mapped shorelines as a gpd.GeoDataFrame    

    Arguments:
    -----------
    output: dict
        contains the coordinates of the mapped shorelines + attributes          
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame
        contains the shorelines + attirbutes
  
    """    
     
    # loop through the mapped shorelines
    counter = 0
    for i in range(len(output['shorelines'])):
        # skip if there shoreline is empty 
        if len(output['shorelines'][i]) == 0:
            continue
        else:
            # save the geometry + attributes
            geom = geometry.LineString(output['shorelines'][i])
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
            gdf.index = [i]
            gdf.loc[i,'date'] = output['dates'][i].strftime('%Y-%m-%d %H:%M:%S')
            gdf.loc[i,'satname'] = output['satname'][i]
            gdf.loc[i,'geoaccuracy'] = output['geoaccuracy'][i]
            gdf.loc[i,'cloud_cover'] = output['cloud_cover'][i]
            # store into geodataframe
            if counter == 0:
                gdf_all = gdf
            else:
                gdf_all = gdf_all.append(gdf)
            counter = counter + 1
            
    return gdf_all

def transects_to_gdf(transects):
    """
    Saves the shore-normal transects as a gpd.GeoDataFrame    

    Arguments:
    -----------
    transects: dict
        contains the coordinates of the transects          
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame

        
    """  
       
    # loop through the mapped shorelines
    for i,key in enumerate(list(transects.keys())):
        # save the geometry + attributes
        geom = geometry.LineString(transects[key])
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
        gdf.index = [i]
        gdf.loc[i,'name'] = key
        # store into geodataframe
        if i == 0:
            gdf_all = gdf
        else:
            gdf_all = gdf_all.append(gdf)
            
    return gdf_all


###################################################################################################
# IMAGE ANNOTATOR
###################################################################################################

def annotate_images(source=None, inputs=None, 
                    label_type='polygon', output_epsg=4326, 
                    cloud_mask_issue=False):
    """
    Allows the user to manually draw polygons to annotate images stored in metadata
    from download (works only with RGB images) or in any chosen directory (works also
    with grayscale / 1-channel images).

    Arguments:
    -----------
    source: dict or str
        either metadata dict from sat_download, or a chosen folder path
    inputs: dict (optional, only necessary if source=metadata)
        input parameters from sat_download (sitename, filepath, polygon, dates, sat_list)
    label_type: str
        type of annotations - either 'polygon' or 'binary'    
    cloud_mask_issue: boolean, optional (default: False)
        True if there is an issue with the cloud mask and pixels are erroneously
        being masked on the images
    output_epsg: int, optional (default: 4326)
        output spatial reference system as EPSG code

    Returns (either one or the other, depending on label_type):
    -----------
    pts_world_final: np.array - if label_type == 'polygon'
        world coordinates of the polygons that were manually digitized - also saved as 
        .pkl and .geojson files
    binary_dict: dict - if label_type == 'binary'
        dictionary with image names as keys and the corresponding True/False labels as values
    """
    
    # determine if the source is metadata or a custom folder
    if type(source) == dict:
        
        # check if annotations already exist in the folder corresponding to metadata
        sitename = inputs['sitename']
        filepath_data = inputs['filepath']
        filepath = os.path.join(filepath_data, sitename)
        filename = sitename + '_polygon_annotations.pkl'
        filename_bin = sitename + '_binary_annotations.csv'
        if filename in os.listdir(filepath):
            print('Annotations already exist and have been loaded.')
            with open(os.path.join(filepath, filename), 'rb') as f:
                polygon_annotations = pickle.load(f)
            return polygon_annotations
        if filename_bin in os.listdir(filepath):
            print('Annotations already exist and have been loaded.')
            binary_dict = pd.read_csv(os.path.join(source, filename_bin),
                                      header=None, names=['image_name','label'])
            return binary_dict

        # otherwise get the list of images to digitize (only S2, L8 or L5 images, no L7 because of scan line error)
        else:
            metadata = source
            # first try to use S2 images (10m res)
            if 'S2' in metadata.keys():
                satname = 'S2'
                filepath = sat_tools.get_filepath(inputs,satname)
                filenames = metadata[satname]['filenames']
            # if no S2 images, try L8 (15m res in the RGB with pansharpening)
            elif not 'S2' in metadata.keys() and 'L8' in metadata.keys():
                satname = 'L8'
                filepath = sat_tools.get_filepath(inputs,satname)
                filenames = metadata[satname]['filenames']
            # if no S2 images and no L8, use L5 images
            elif not 'S2' in metadata.keys() and not 'L8' in metadata.keys() and 'L5' in metadata.keys():
                satname = 'L5'
                filepath = sat_tools.get_filepath(inputs,satname)
                filenames = metadata[satname]['filenames']
            else:
                raise Exception('You cannot annotate L7 images (because of gaps in the images),' +\
                                'please add S2, L8 or L5 images to your dataset.')
    elif type(source) == str:
        # check if annotations already exist in source folder, otherwise proceed to drawing
        filenames = os.listdir(source)
        filenames = [file for file in filenames if file.endswith('.tif')]
        if len(filenames) == 0:
            raise Exception('Please make sure your images are georeferenced (.tif).')
        
        # check if there is already a labels file
        sitename = os.path.basename(os.path.normpath(source))
        if label_type == 'polygon' and os.path.isfile(os.path.join(source, sitename + '_polygon_annotations.pkl')):
            print('Annotations already exist and have been loaded.')
            with open(os.path.join(source, sitename + '_polygon_annotations.pkl'), 'rb') as f:
                polygon_annotations = pickle.load(f)
            return polygon_annotations
        if label_type == 'binary' and os.path.isfile(os.path.join(source, sitename + '_binary_annotations.csv')):
            print('Annotations already exist and have been loaded.')
            binary_dict = pd.read_csv(os.path.join(source, sitename + '_binary_annotations.csv'),
                                      header=None, names=['image_name','label'])
            return binary_dict
    else:
        raise Exception('Please pass a valid source of images - either metadata dict from Sentinel' +\
                        'or Landsat download, or a folder path.')

    # create figure
    fig, ax = plt.subplots(1,1, figsize=[8,8], tight_layout=True)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    # loop through the images
    pts_pix_final = []
    pts_world_final = []
    geoms_final = []
    gdf_final = []
    binary_dict = {}
    for i in range(len(filenames)):

        # if source is metadata, read and preprocess image
        if type(source) == dict:
            fn = sat_tools.get_filenames(filenames[i], filepath, satname)
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata =\
                            sat_preprocess.preprocess_single(fn, satname, cloud_mask_issue)
            im_RGB = sat_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            image_epsg = metadata[satname]['epsg'][i]

        # if source is folder, read the image and get georef
        else:
            im_RGB = plt.imread(os.path.join(source, filenames[i]))
            try:
                im = gdal.Open(os.path.join(source, filenames[i]), gdal.GA_ReadOnly)
                georef = np.array(im.GetGeoTransform())
                image_epsg = int(gdal.Info(im, format='json')['coordinateSystem']['wkt'].rsplit('"EPSG","', 1)[-1].split('"')[0])
            except:
                raise Exception('Please use only georeferenced images - so that polygons can be' +\
                                'converted to world coordinates')
            if len(im_RGB.shape) >= 3: # RGB image
                im_RGB = im_RGB[...,[0,1,2]]
            else: # grayscale image (same name though)
                im_RGB = im_RGB[...,0]

        # plot the image on a figure
        ax.axis('off')
        ax.imshow(im_RGB)

        # decide if the image if good enough to label it
        ax.set_title('Image {}/{}\n'.format(i+1,len(filenames)) +
                     'Press <right arrow> on your keyboard if the image is clear enough to label it.\n' +
                     'If the image is too cloudy press <left arrow> to get another image', fontsize=14)
        # set a key event (skip_image) to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immutable so we can access it after the keypress event
        skip_image = False
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # Let the user press a key, right arrow to keep the image, left arrow to skip it -
        # the loop runs until a key is pressed, which triggers a 'break'.
        # To break the loop the user can also press 'escape'.
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()

            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()

            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled process.')
            else:
                plt.waitforbuttonpress()
         
        # if skip_image was set to True by keypress, go to next iteration (next image)
        if skip_image:
            ax.clear()
            continue
        
        # else, go to drawing
        else:
            #========================================================================================#
            # Binary annotations
            #========================================================================================#
            if label_type == 'binary':
                # create two new buttons
                true_button = plt.text(0, 0.9, '<True>', size=16, ha="left", va="top",
                                       transform=plt.gca().transAxes,
                                       bbox=dict(boxstyle="square", ec='k',fc='w'))
                false_button = plt.text(1, 0.9, '<False>', size=16, ha="right", va="top",
                                        transform=plt.gca().transAxes,
                                        bbox=dict(boxstyle="square", ec='k',fc='w'))
                while 1:
                    true_button.set_visible(True)
                    false_button.set_visible(True)

                    # update title (instructions)
                    ax.set_title('Click on <True> if the object is present in the image,\n' +
                                 "or on <False> if it isn't.", fontsize=14)
                    plt.draw()

                    # let the user click
                    pt_input = ginput(n=1, timeout=1e9, show_clicks=False)
                    pt_input = np.array(pt_input)
                    if pt_input[0][0] > im_RGB.shape[1]/2: # if click on <False> (right of the image)
                        binary = False
                    else: 
                        binary = True

                    # ask for confirmation
                    true_button.set_visible(False)
                    false_button.set_visible(False)
                    ax.set_title('{}\nTo confirm, press <enter> on your keyboard.'.format(binary) +\
                                 ' To restart, press <esc>.', fontsize=14)
                    plt.draw()

                    key_event = {}
                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.waitforbuttonpress()
                    if key_event.get('pressed') == 'enter':
                        # save the label
                        binary_dict[filenames[i]] = binary

                        # if we are at the end of the images, display final message
                        if not i+1 < len(filenames):
                            plt.draw()
                            plt.title('All labels saved as ' + sitename + '_binary_annotations.csv', fontsize=14)
                            ginput(n=1, timeout=3, show_clicks=False)
                            break
                        # if there are other images left, display information for the user
                        else:
                            plt.draw()
                            plt.title('Label saved. Going to next image...', fontsize=14)
                            ginput(n=1, timeout=1, show_clicks=False)
                            break

                    # if <esc>, restart this 'while' iteration to get another chance to label
                    elif key_event.get('pressed') == 'escape':
                        continue
                    else:
                        plt.waitforbuttonpress()
            
                # finally, go to next picture, or close GUI (if last iteration)
                if i+1 < len(filenames):
                    ax.clear()
                    continue
                else:
                    # save labels to disk
                    if type(source)==dict:
                        filepath = os.path.join(filepath_data, sitename)
                    else:
                        filepath = source
                    with open(os.path.join(filepath, sitename + '_binary_annotations.csv'),'w') as f:
                        w = csv.writer(f)
                        w.writerows(binary_dict.items())
                        print('Binary annotations have been saved in {}.'.format(filepath))
                    # close interface and return labels dictionary
                    plt.close()
                    return binary_dict

            #========================================================================================#
            # Polygon annotations
            #========================================================================================#
            elif label_type == 'polygon':
                # create two new buttons
                add_button = plt.text(0, 0.9, 'add', size=16, ha="left", va="top",
                                       transform=plt.gca().transAxes,
                                       bbox=dict(boxstyle="square", ec='k',fc='w'))
                end_button = plt.text(1, 0.9, 'end', size=16, ha="right", va="top",
                                       transform=plt.gca().transAxes,
                                       bbox=dict(boxstyle="square", ec='k',fc='w'))
                
                # add multiple polygons (until user clicks on <end> button)
                pts_world_image = []
                pts_pix_image = []
                geoms = []
                while 1:
                    add_button.set_visible(False)
                    end_button.set_visible(False)
                    
                    # update title (instructions)
                    ax.set_title('Click points to capture the polygon shape.\n' +
                                 'When finished digitizing, press <ENTER>', fontsize=14)
                    plt.draw()

                    # let user click on the polygon summits
                    pts = ginput(n=50000, timeout=1e9, show_clicks=True) # max 50k points per polygon!
                    pts_pix = np.array(pts)
                    #if pts_pix.shape[1] == 3:
                    #    pts_pix = np.delete(pts_pix,2, axis=1)

                    # convert pixel coordinates to world coordinates
                    pts_world = sat_tools.convert_pix2world(pts_pix[:,[1,0]], georef)
                    #if pts_world.shape[1] == 3:
                    #    pts_world = np.delete(pts_world, 2, axis=1)

                    # save as geometry (to create .geojson file later) and as array
                    geoms.append(geometry.Polygon(pts_world[:-1])) 
                    pts_world_image.append(pts_world[:-1])
                    pts_pix_image.append(pts_pix[:-1])

                    # interpolate between points clicked by the user (1m resolution) for visualization
                    pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                    for k in range(len(pts_world[:-1])):
                        pt_dist = np.linalg.norm(pts_world[k,:]-pts_world[k+1,:])
                        xvals = np.arange(0,pt_dist)
                        yvals = np.zeros(len(xvals))
                        pt_coords = np.zeros((len(xvals),2))
                        pt_coords[:,0] = xvals
                        pt_coords[:,1] = yvals
                        phi = 0
                        deltax = pts_world[k+1,0] - pts_world[k,0]
                        deltay = pts_world[k+1,1] - pts_world[k,1]
                        phi = np.pi/2 - np.math.atan2(deltax, deltay)
                        tf = transform.EuclideanTransform(rotation=phi, translation=pts_world[k,:])
                        pts_world_interp = np.append(pts_world_interp,tf(pt_coords), axis=0)
                    pts_world_interp = np.delete(pts_world_interp,0,axis=0)
                    pts_pix_interp = sat_tools.convert_world2pix(pts_world_interp, georef)
                    ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--')
                    #ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko')
                    #ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko')

                    # update title and buttons
                    add_button.set_visible(True)
                    end_button.set_visible(True)
                    ax.set_title('Click on <add> to digitize another polygon in this image, or on <end> to save.',
                              fontsize=14)
                    plt.draw()

                    # let the user click again (<add> or <end>)
                    pt_input = ginput(n=1, timeout=1e9, show_clicks=False)
                    pt_input = np.array(pt_input)

                    # if user clicks on <end>, break the loop and go to saving (if <add>, go on to new iteration)
                    if pt_input[0][0] > im_RGB.shape[1]/2:
                        add_button.set_visible(False)
                        end_button.set_visible(False)
                        if i+1 < len(filenames):
                            plt.draw()
                            plt.title('Polygon(s) saved. Going to next image...', fontsize=14)
                            ginput(n=1, timeout=1, show_clicks=False)
                        else:
                            plt.draw()
                            plt.title('All polygon(s) saved as ' + sitename + '_polygon_annotations.pkl\nand ' +\
                                      sitename + '_polygon_annotations.geojson', fontsize=14)
                            ginput(n=1, timeout=3, show_clicks=False)
                        break

            # now we finished getting coordinates for the image, just convert to custom crs
            pts_world_image = sat_tools.convert_epsg(pts_world_image, image_epsg, output_epsg)
            # the line above outputs 3D coords, don't kown why, but whatever, it still works

            # create dataframe with information about polygon geometries in order to write geojson later
            for k, poly in enumerate(geoms):
                # store pixel coords, world coords and image name into geodataframe
                gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(poly))
                gdf.index = [k]
                gdf.loc[k,'filename'] = filenames[i]
                pts_json = np.asarray(pts_pix_image[k]).tolist()
                gdf.loc[k,'pixel_geometry'] = json.dumps(pts_json)
                if k == 0: # need to initialise gdf at first iteration
                    gdf_image = gdf
                else:
                    gdf_image = gdf_image.append(gdf)
            # set geodataframe crs to image_epsg and convert to user-defined crs
            gdf_image.crs = {'init':'epsg:'+str(image_epsg)}
            gdf_image = gdf_image.to_crs({'init': 'epsg:'+str(output_epsg)})

            # add coordinates in pixels, coordinates in world crs, geometries and geodataframe to final lists
            pts_pix_final.extend(pts_pix_image)
            pts_world_final.extend(pts_world_image)
            geoms_final.extend(geoms)
            try:
                test = len(gdf_final)
                need_to_initialize_gdf_final = True
            except:
                need_to_initialize_gdf_final = False
            if need_to_initialize_gdf_final:
                gdf_final = gdf_image
            else:
                gdf_final = gdf_final.append(gdf_image)

            # finally, go to next picture, or close GUI (if last iteration)
            if i+1 < len(filenames):
                ax.clear()
                continue
            else:
                plt.close()

            # if we arrived here, it means we are in the last iteration (last image)!
            # save final list of coordinates in world crs as .pkl
            if type(source)==dict:
                filepath = os.path.join(filepath_data, sitename)
            else:
                filepath = source
            with open(os.path.join(filepath, sitename + '_polygon_annotations.pkl'), 'wb') as f:
                pickle.dump(pts_world_final, f)

            # save final geodataframe with all information as .geojson
            gdf_final.to_file(os.path.join(filepath, sitename + '_polygon_annotations.geojson'),
                              driver='GeoJSON', encoding='utf-8')

            print('Polygon annotations have been saved in {}.'.format(filepath))
            break
    
    # if the last image was skipped, save work
    if skip_image:
        plt.close()
        if type(source)==dict:
                filepath = os.path.join(filepath_data, sitename)
        else:
            filepath = source
        with open(os.path.join(filepath, sitename + '_polygon_annotations.pkl'), 'wb') as f:
            pickle.dump(pts_world_final, f)
        # save final geodataframe with all information as .geojson
        gdf_final.to_file(os.path.join(filepath, sitename + '_polygon_annotations.geojson'),
                          driver='GeoJSON', encoding='utf-8')
        print('Polygon annotations have been saved in {}.'.format(filepath))

    # check if a polygon has been digitised
    if len(pts_world_final) == 0:
        raise Exception('No cloud free images are available to draw polygons,'+
                        'download more images and try again')

    return pts_world_final


###################################################################################################
# IMAGE SLICER
###################################################################################################

"""
This sub-module contains functions to split images in a selected number of tiles (and join them
back together). This split doesn't conserve metadata (in particular georeferenced information).
To save small georeferenced tiles, use the `tile_size` argument in the sat_dowload functions.

Original author: Sam Dobson, London, 2018
                 https://github.com/samdobson/image_slicer
Modifications and additions: Thomas de Mareuil, Total E-LAB, 2020
"""

from math import sqrt, ceil, floor
from PIL import Image

class Tile(object):
    """Represents a single tile."""

    def __init__(self, image, number, position, coords, filename=None):
        self.image = image
        self.number = number
        self.position = position
        self.coords = coords
        self.filename = filename

    @property
    def row(self):
        return self.position[0]

    @property
    def column(self):
        return self.position[1]

    @property
    def basename(self):
        """Strip path and extension. Return base filename."""
        return get_basename(self.filename)

    def generate_filename(self, directory=os.getcwd(), prefix='tile',
                          format='png', path=True):
        """Construct and return a filename for this tile."""
        filename = prefix + '_{col:02d}_{row:02d}.{ext}'.format(
                      col=self.column, row=self.row, ext=format.lower().replace('jpeg', 'jpg'))
        if not path:
            return filename
        return os.path.join(directory, filename)

    def save(self, filename=None, format='png'):
        if not filename:
            filename = self.generate_filename(format=format)
        self.image.save(filename, format)
        self.filename = filename

    def __repr__(self):
        """Show tile number, and if saved to disk, filename."""
        if self.filename:
            return '<Tile #{} - {}>'.format(self.number,
                                            os.path.basename(self.filename))
        return '<Tile #{}>'.format(self.number)


def calc_columns_rows(n):
    """
    Calculate the number of columns and rows required to divide an image
    into ``n`` parts.

    Return a tuple of integers in the format (num_columns, num_rows)
    """
    num_columns = int(ceil(sqrt(n)))
    num_rows = int(ceil(n / float(num_columns)))
    return (num_columns, num_rows)


def get_combined_size(tiles):
    """Calculate combined size of tiles."""
    # TODO: Refactor calculating layout to avoid repetition.
    columns, rows = calc_columns_rows(len(tiles))
    tile_size = tiles[0].image.size
    return (tile_size[0] * columns, tile_size[1] * rows)


def join(tiles, width=0, height=0):
    """
    @param ``tiles`` - Tuple of ``Image`` instances.
    @param ``width`` - Optional, width of combined image.
    @param ``height`` - Optional, height of combined image.
    @return ``Image`` instance.
    """
    # Don't calculate size if width and height are provided
    # this allows an application that knows what the
    # combined size should be to construct an image when
    # pieces are missing.

    if width > 0 and height > 0:
        im = Image.new('RGB',(width, height), None)
    else:
        im = Image.new('RGB', get_combined_size(tiles), None)
    columns, rows = calc_columns_rows(len(tiles))
    for tile in tiles:
        try:
            im.paste(tile.image, tile.coords)
        except IOError:
            #do nothing, blank out the image
            continue
    return im


def validate_image(image, number_tiles):
    """Basic sanity checks prior to performing a split."""
    TILE_LIMIT = 99 * 99

    try:
        number_tiles = int(number_tiles)
    except:
        raise ValueError('number_tiles could not be cast to integer.')

    if number_tiles > TILE_LIMIT or number_tiles < 2:
        raise ValueError('Number of tiles must be between 2 and {} (you \
                          asked for {}).'.format(TILE_LIMIT, number_tiles))


def validate_image_col_row(image , col , row):
    """Basic checks for columns and rows values"""
    SPLIT_LIMIT = 99

    try:
        col = int(col)
        row = int(row)
    except:
        raise ValueError('columns and rows values could not be cast to integer.')

    if col < 2:
        raise ValueError('Number of columns must be between 2 and {} (you \
                          asked for {}).'.format(SPLIT_LIMIT, col))
    if row < 2 :
        raise ValueError('Number of rows must be between 2 and {} (you \
                          asked for {}).'.format(SPLIT_LIMIT, row))


def slice_image(filename, number_tiles=None, col=None, row=None, max_tile_size=None):
    """
    Split an image into a specified number of tiles.

    Args:
       filename (str):  The filename of the image to split.
       number_tiles (int):  The number of tiles required.
       col, row (int): Nb of columns required (alternative to number_tiles)
       max_tile_size (int*int): Max tile dimensions.

    Returns:
        Tuple of :class:`Tile` instances.
        Tuple with tile dimensions.
    """
    im = Image.open(filename)
    im_w, im_h = im.size

    columns = 0
    rows = 0
    if not number_tiles is None:
        validate_image(im, number_tiles)
        columns, rows = calc_columns_rows(number_tiles)
    #    extras = (columns * rows) - number_tiles
        tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))
    else:
        validate_image_col_row(im, col, row)
        columns = col
        rows = row
    #    extras = (columns * rows) - number_tiles
        tile_w, tile_h = int(sqrt(max_tile_size)), int(sqrt(max_tile_size))

    tiles = []
    number = 1
    for pos_y in range(0, im_h - rows, tile_h): # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w): # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            image = im.crop(area)
            position = (int(floor(pos_x / tile_w)) + 1,
                        int(floor(pos_y / tile_h)) + 1)
            coords = (pos_x, pos_y)
            tile = Tile(image, number, position, coords)
            tiles.append(tile)
            number += 1
    print('Splitting the original image in {} tiles'.format(len(tiles)),
          'of size {}x{}...'.format(tile_w, tile_h))

    return tuple(tiles), (tile_w, tile_h)


def save_tiles(tiles, prefix='', directory=os.getcwd(), format='png'):
    """
    Write image files to disk. Create specified folder(s) if they
       don't exist. Return list of :class:`Tile` instance.

    Args:
       tiles (list):  List, tuple or set of :class:`Tile` objects to save.
       prefix (str):  Filename prefix of saved tiles.
       directory (str):  Directory to save tiles
       format (str): Image format to write.

    Returns:
        Tuple of :class:`Tile` instances.
    """

    for tile in tiles:
        tile.save(filename=tile.generate_filename(prefix=prefix,
                                                  directory=directory,
                                                  format=format),
                                                  format=format)
    return tuple(tiles)


def get_image_column_row(filename):
    """Determine column and row position for filename."""
    row, column = os.path.splitext(filename)[0][-5:].split('_')
    return (int(column)-1, int(row)-1)


def open_images_in(directory):
    """Open all images in a directory. Return tuple of Tile instances."""

    files = [filename for filename in os.listdir(directory)
                    if '_' in filename and not filename.startswith('joined')]
    tiles = []
    if len(files) > 0:
        i = 0
        for file in files:
            pos = get_image_column_row(file)
            im = Image.open(os.path.join(directory, file))

            position_xy=[0,0]
            count=0
            for a,b in zip(pos,im.size):
                position_xy[count] = a*b
                count = count + 1
            tiles.append(Tile(image=im, position=pos, number=i+1,
                                coords=position_xy, filename=file))
            i = i + 1
    return tiles


'''
Helper functions for ``image_slicer``.
'''
import os
from PIL import Image


def get_basename(filename):
    """Strip path and extension. Return basename."""
    return os.path.splitext(os.path.basename(filename))[0]


def open_images(directory):
    """Open all images in a directory. Return tuple of Image instances."""
    return [Image.open(os.path.join(directory, file)) for file in os.listdir(directory)]


def get_columns_rows(filenames):
    """Derive number of columns and rows from filenames."""
    tiles = []
    for filename in filenames:
        row, column = os.path.splitext(filename)[0][-5:].split('_')
        tiles.append((int(row), int(column)))
    rows = [pos[0] for pos in tiles]; columns = [pos[1] for pos in tiles]
    num_rows = max(rows); num_columns = max(columns)
    return (num_columns, num_rows)


def split_image(image_path, filepath_out=None, max_tile_size=512*512, square=False,
                                                nb_tiles=None, inputs=None, show=False):
    """
    Split an image in a nb of tiles such that all tiles are under 512x512 pixels.
    
    Arguments:
    -----------
    image_path: str
        path to original image downloaded through the sat_download module
    filepath_out: str, optional (default: new folder named /preprocessed_files/split_tiles)
        path to directory to save split image tiles
    max_tile_size: int, optional (default: 512*512)
        maximum size of the final tiles, in pixels
    square: bool, optional (default: False)
        forces the tiles to a square shape - if False, keeps the aspect ratio
        of the original image (and tile surface smaller than default 512*512)
    nb_tiles: int, optional (default: None, max: 99x99)
        force the number of tiles (size wil adjust accordingly)
    inputs: dict, optional (if not, need to pass filepath_out)
        dict with the sat_download inputs to get default folder path
    show: bool, optional (default: False)
        output a visualization of the original image with the split grid on top

    Returns:
    -----------
    image tiles: .png images
        as many tiles of requested size as the original image can fit
    """

    # if no output path was provided, set to default
    if filepath_out == None:
        assert inputs != None, ('Please provide a filepath_out or the preprocessing' +
                                                    ' `inputs` dictionary with path inside.')
        sitename = inputs['sitename']
        filepath_data = inputs['filepath']
        filepath_out = os.path.join(filepath_data, sitename, 'preprocessed_files', 'split_tiles')
        if not os.path.exists(filepath_out):
            os.makedirs(filepath_out)

    # open original image
    im = Image.open(image_path)
    im_w, im_h = im.size
    
    # compute tiles
    if nb_tiles != None:
        tiles, tile_dim = slice_image(image_path, number_tiles=nb_tiles, save=False)

    else:
        if not square:
            nb_tiles = (im_w * im_h) // max_tile_size
            tiles, tile_dim = slice_image(image_path, number_tiles=nb_tiles, save=False)
        
        if square:
            col = int(round(im_w // sqrt(max_tile_size),0))
            row = int(round(im_h // sqrt(max_tile_size),0))
            if im_w % sqrt(max_tile_size) != 0 or im_h % sqrt(max_tile_size) != 0:
                print("Warning: tile dimensions don't exactly match with image",
                      "- some tiles might have black padding to compensate.")
            tiles, tile_dim = slice_image(image_path, col=col, row=row, max_tile_size=max_tile_size)
    
    # save tiles
    save_tiles(tiles, directory=filepath_out, prefix=get_basename(image_path), format='png')
    print('Tiles successfully saved in {}.'.format(filepath_out))

    # show grid
    if show:
        grid_x_ticks = np.arange(0, im_w, tile_dim[0])
        grid_y_ticks = np.arange(0, im_h, tile_dim[1])
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(im)
        plt.title('Split preview')
        ax.set_xticks(grid_x_ticks , minor=True)
        ax.set_yticks(grid_y_ticks , minor=True)
        ax.grid(which='minor', linestyle='--', color='black', linewidth=2)
        plt.show()
