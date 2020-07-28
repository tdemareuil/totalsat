"""
This module contains functions to download satellite images (Landsat 
5-7-8 and Sentinel 2) from the Google Earth Engine python API
and to merge duplicates using GDAL.

Original author: Kilian Vos, Water Research Laboratory,
                 University of New South Wales, 2018
                 https://github.com/kvos/CoastSat
Modifications and additions: Thomas de Mareuil, Total E-LAB, 2020
"""

# load earth engine-related modules
import ee
from urllib.request import urlretrieve
import zipfile
import copy

# image processing modules
from skimage import morphology, transform
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas as gpd
import folium

# additional modules
import os
import pdb
from datetime import datetime, timedelta
import pytz
import pickle
import math
import osmnx
from shapely.ops import transform, cascaded_union
import pyproj

# other totalsat modules
from totalsat import sat_preprocess, sat_tools

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans


###################################################################################################
# DOWNLOAD IMAGES
###################################################################################################

def check_images_available(inputs):
    """
    Check image availity from GEE for each specified satellite mission,
    with chosen parameters to filter images.
      
    Arguments:
    -----------
    inputs: dict with the following keys
        'sat_list': list of str
            list that contains the names of the satellite missions to include:
            ```
            sat_list = ['L5', 'L7', 'L8', 'S2', 'S2_RGB']
            ```
            S2_RGB is the collection with the least images (starting in April 2017), but the
            advantage is that it provides directly true color RGB images (no preprocessing needed!)
        'include_T2': bool, optional (default: False)
            include Landsat T2 collections in image search
        'filepath': str
            filepath to the directory where the images will be downloaded
        'sitename': str, optional (default: current date)
            name of the site / project, which will also be the name of the created download folder
        'max_could_cover': int, optional (default: 10)
            maximum cloud cover percentage to be passed to the remove_cloudy_images
            filtering function
        'polygon': list
            list containing the pairs of lon/lat coordinates of the polygon area to be 
            extracted, longitudes in the 1st column and latitudes in the 2nd column.
            Max area size is 100x100 km, and format should be as below:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8]]]
            ```
        'polygons_list': list of lists, optional (default: None)
            list of polygons as coordinates list (to search for several polygon areas at once)
        'point': list
            latitude and longitude of the point at the center of the area to be 
            extracted - user should input either polygon or point + radius, in format:
            ```
            point = [lon, lat]
            ```
        'radius': float, optional (default: 20)
            radius of the area to be extracted around the point coordinates, in km (max:50)
        'dates': list of str, optional (default: last image available)
            either 'last' or a list that contains 2 strings, with the initial and final
            dates in format 'yyyy-mm-dd' (the final date can also be 'today'):
            ```
            dates = ['1987-01-01', '2018-01-18']
            ```
        'unique': bool, optional (default = False)
            set to True if you want to select a unique image (the last one) for the specified dates
        'max_size': int, optional (default: None)
            maximum image size (width), in pixels - if the area of interest is too large,
            it will be split in several images of max_size
        'merge': bool, optional (default: True)
            set to False if you don't want to merge images taken at the same date (for ex
            if you download several images from the same S2 tile, taken the same day - 
            therefore automatically set to true if inputs include polygons_list)
    
    Returns:
    -----------
    im_dict_T1: dict
        list of images in Landsat Tier 1 and Sentinel-2 Level-1C / Level 2-A
    im_dict_T2: dict
        list of images in Tier 2 (Landsat only)
    # TODO - add pre-computed 8-day composite images:
    # im_dict_composites: dict
    #     list of pre-computed composite Landsat images
    """
    
    # initializations
    ee.Initialize()
    collection_names = {'L5':'LANDSAT/LT05/C01/T1_TOA',
                        'L7':'LANDSAT/LE07/C01/T1_TOA',
                        'L8':'LANDSAT/LC08/C01/T1_TOA',
                        'S2':'COPERNICUS/S2',
                        'S2_RGB':'COPERNICUS/S2_SR'#,
                        #TODO - add pre-computed 8-day composite images:
                        #'L8_BAI': 'LANDSAT/LC08/C01/T1_8DAY_BAI',
                        #'L8_EVI': 'LANDSAT/LC08/C01/T1_8DAY_EVI',
                        #'L8_NDVI': 'LANDSAT/LC08/C01/T1_8DAY_NDVI',
                        #'L8_NBRT': 'LANDSAT/LC08/C01/T1_8DAY_NBRT',
                        #'L8_NDSI': 'LANDSAT/LC08/C01/T1_8DAY_NDSI',
                        #'L8_NDWI': 'LANDSAT/LC08/C01/T1_8DAY_NDWI'
                        }
    
    # check if dates were provided - if not, set to 'last'
    if ('dates' not in inputs) or (inputs['dates'] == 'last'):
        last = True
        now = datetime.now()
        last_2months = now - timedelta(weeks=8)
        inputs['dates'] = [last_2months.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')]
        print('Looking for the last available data...')
    else:
        last = False
    # if 'today' was provided, replace with current date
    if 'dates' in inputs and 'today' in inputs['dates']: 
        inputs['dates'][1] = datetime.now().strftime('%Y-%m-%d')
    # if 'unique' was set to True, create a variable
    if 'unique' in inputs and inputs['unique'] == True:
        unique = True

    # check if polygon area was provided - if not, define polygon area around point
    if (not 'polygon' in inputs) and (not 'polygons_list' in inputs): # sanity check
        assert 'point' in inputs, 'Please provide a point or polygon coordinates to search for images.'
    if 'point' in inputs:
        if not 'radius' in inputs: # default radius
            inputs['radius'] = 20
        
        # Compute the polygon AOI and split it if requested (i.e. radius too large compared to max_size)
        inputs['polygon'] = sat_tools.bbox_from_point(inputs['point'], inputs['radius'])
        pixel_width = inputs['radius']*2*1000 / 10
        if 'max_size' in inputs and pixel_width > inputs['max_size']:
            inputs['polygons_list'] = sat_tools.split_area(inputs['polygon'], pixel_width, inputs['max_size'])
            print('Your area of interest will be split into smaller image areas to fit max_size' +\
                  ' requirements.\n\nSearching for images on the GEE server...\n')

    # set maximum cloud cover filter if passed as input
    if 'max_cloud_cover' in inputs:
        prc_cloud_cover = inputs['max_cloud_cover']
    else:
        prc_cloud_cover = 10

    # check how many images are available in Landsat Tier 1 and Sentinel Level-1C
    col_names_T1 = {new_key: collection_names[new_key] for new_key in ['L5', 'L7', 'L8', 'S2', 'S2_RGB']}
    print('- In Landsat Tier 1 & Sentinel-2 Level-1C / Level 2-A:')
    im_dict_T1 = dict([])
    im_dict_T2 = dict([])
    sum_img_T1 = 0
    sum_img_T2 = 0
    for satname in inputs['sat_list']:

        # if AOI was split into smaller areas, loop over all polygons to get list of images
        if 'polygons_list' in inputs:
            im_dict_T1[satname] = []
            counter = 0
            for i, polygon_i in enumerate(inputs['polygons_list']):
                # get list of images in GEE collection
                while True:
                    try:
                        ee_col = ee.ImageCollection(col_names_T1[satname])
                        col = ee_col.filterBounds(ee.Geometry.Polygon(polygon_i))\
                                    .filterDate(inputs['dates'][0],inputs['dates'][1])
                        im_list = col.getInfo().get('features')
                        break
                    except:
                        continue
                # remove images above a chosen cloud percentage
                im_list_upt = remove_cloudy_images(im_list, satname, prc_cloud_cover)
                # remove UTM duplicates in S2 collections (they provide several projections for same images)
                if satname == 'S2' and len(im_list_upt)>1: 
                    im_list_upt = filter_S2_collection(im_list_upt)
                # if requested, select only the last image
                if last == True or unique == True:
                    try:
                        im_list_upt = [im_list_upt[-1]]
                    except:
                        print('')
                # add polygon index to each image's metadata
                for k in range(len(im_list_upt)):
                    im_list_upt[k]['polygon_index'] = i
                # add image metadata to list of images and augment counter
                im_dict_T1[satname].extend(im_list_upt)
                if last == True or unique == True:
                    sum_img_T1 += 1
                    counter += 1
                else:
                    sum_img_T1 += len(im_list_upt)
                    counter += len(im_list_upt)
            print('  %s: %d image(s)'%(satname, counter))

        # else, just get list of images in GEE collection
        else:
            while True:
                try:
                    ee_col = ee.ImageCollection(col_names_T1[satname])
                    col = ee_col.filterBounds(ee.Geometry.Polygon(inputs['polygon']))\
                                .filterDate(inputs['dates'][0],inputs['dates'][1])
                    im_list = col.getInfo().get('features')
                    break
                except:
                    continue
            # remove images above a chosen cloud percentage
            im_list_upt = remove_cloudy_images(im_list, satname, prc_cloud_cover)
            # remove UTM duplicates in S2 collections (they provide several projections for same images)
            if satname == 'S2' and len(im_list_upt)>1: 
                im_list_upt = filter_S2_collection(im_list_upt)
            # if requested, select only the last image
            if last == True or unique == True:
                try:
                    im_list_upt = [im_list_upt[-1]]
                except:
                    print('')
            sum_img_T1 += len(im_list_upt)
            print('  %s: %d image(s)'%(satname, len(im_list_upt)))
            im_dict_T1[satname] = im_list_upt
    
    # if requested, also check Landsat Tier 2 collections
    # TODO: rewrite this section with similar structure as section above (polygons_split, etc.)
    if ('include_T2' in inputs) and (inputs['include_T2'] == True):
        col_names_T2 = {new_key: collection_names[new_key] for new_key in ['L5', 'L7', 'L8']}
        print('- In Landsat Tier 2:', end='\n')
        for satname in inputs['sat_list']:
            if satname == 'S2': continue
            # get list of images in GEE collection
            while True:
                try:
                    ee_col = ee.ImageCollection(col_names_T2[satname])
                    col = ee_col.filterBounds(ee.Geometry.Polygon(inputs['polygon']))\
                                .filterDate(inputs['dates'][0],inputs['dates'][1])
                    if last == True or unique == True:
                        col = col.limit(1, 'system:time_start', False)
                    im_list = col.getInfo().get('features')
                    break
                except:
                    continue
            # remove cloudy images
            im_list_upt = remove_cloudy_images(im_list, satname, prc_cloud_cover)
            # if requested, select only the last image
            if last == True or unique == True:
                im_list_upt = [im_list_upt[-1]]
            sum_img_T2 += len(im_list_upt)
            print('  %s: %d image(s)'%(satname,len(im_list_upt)))
            im_dict_T2[satname] = im_list_upt

    # display total
    if last == True and (sum_img_T1+sum_img_T2) != 0:
        inputs['dates'] = 'last' # set dates back to 'last'
        print('  Total: %d image(s) selected.'%(sum_img_T1+sum_img_T2))
    elif last == True and (sum_img_T1+sum_img_T2) == 0:
        inputs['dates'] = 'last' # set dates back to 'last'
        print('\nNo images found in the past 8 weeks with these criteria' +
              ' - to get the last\nimage available, either allow more clouds' +
              ' or search\nfor a longer period explicitly.')
    else:
        print('  Total: %d image(s) selected.'%(sum_img_T1+sum_img_T2))

    return im_dict_T1, im_dict_T2


def plot_requested_area(inputs):
    """
    Plots the requested areas to download on an interactive folium map, for 
    visualization.

    Arguments:
    -----------
    inputs: dict 
        inputs dictionary with parameters for images to select - see 
        detailed list of possible keys in function check_images_available
    
    Returns:
    -----------
        folium map with polygon area(s) as map overlay(s).
    """

    # Compute polygon area(s) if not provided explicitly or computed before
    if ('point' in inputs) and ('polygon' not in inputs):
        if not 'radius' in inputs:
            inputs['radius'] = 20
        
        inputs['polygon'] = sat_tools.bbox_from_point(inputs['point'], inputs['radius'])
        pixel_width = inputs['radius']*2*1000 / 10
        
        if 'max_size' in inputs and pixel_width > inputs['max_size']:
            inputs['polygons_list'] = sat_tools.split_area(inputs['polygon'], pixel_width, inputs['max_size'])

    # Convert large and small polygon to geodataframe for plotting
    crs = {'init': 'epsg:4326'}
    polygon_shp_large = Polygon(inputs['polygon'])
    polygon_df_large = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_shp_large])

    if 'polygons_list' in inputs:
        polygons_shp = [Polygon(coords) for coords in inputs['polygons_list']]
        polygons_df = gpd.GeoDataFrame(crs=crs)
        polygons_df['geometry'] = None
        for i in range(len(polygons_shp)):
            polygons_df.loc[i, 'geometry'] = polygons_shp[i]             

    # Plot polygon area(s) on an interactive map
    m = folium.Map([inputs['polygon'][0][1], inputs['polygon'][0][0]], zoom_start=9)
    folium.GeoJson(polygon_df_large, name='requested area').add_to(m) # plot large area
    if 'polygons_list' in inputs:
        folium.GeoJson(polygons_df, name='split area',
                   style_function=lambda x: {'color':'#228B22'}).add_to(m) # plot split areas

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


def retrieve_images(inputs, im_dict_T1=None):
    """
    Downloads all images from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2
    covering the area of interest, acquired between the specified dates and with
    the specified maximum cloud cover.

    The downloaded images are in .tif format and organised in subfolders, divided
    by satellite mission and subdivided by pixel resolution.

    For each image, several bands are downloaded and stored in a single file (for
    S2-10m the R/G/B/NIR bands, for S2-20m the SWIR band, for S2-60m the Quality
    Assessment band, and for Landsat 7 & 8 the same + the Coastal Aerosols band,
    and the panchromatic band which is stored separately). GDAL's `GetRasterBand` 
    method then splits the various bands for computations during preprocessing.

    Arguments:
    -----------
    inputs: dict 
        inputs dictionary with parameters for images to select - see 
        detailed list of possible keys in function check_images_available
    im_dict_T1: dict, optional (default: None)
        you can pass as argument the output of check_images_available(), in order
        to avoid re-checking available images online

    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system
    """
    
    # initialise connection with GEE server
    ee.Initialize()
    
    # check image availabiliy and retrieve list of images
    try:
        test = len(im_dict_T1)
    except:
        im_dict_T1, im_dict_T2 = check_images_available(inputs)
    
    # if user also wants to download T2 images, merge both lists
    if ('include_T2' in inputs.keys()) and (inputs['include_T2'] == True):
        for key in inputs['sat_list']:
            if key == 'S2': continue
            else: im_dict_T1[key] += im_dict_T2[key]
        
    # create a new directory name for this site of study
    if not 'sitename' in inputs:
        inputs['sitename'] = 'Data-{}'.format(datetime.now().strftime('%Y-%m-%d'))
    im_folder = os.path.join(inputs['filepath'],inputs['sitename'])
    if not os.path.exists(im_folder): os.makedirs(im_folder)    

    # Download images listed in im_dict_T1, with polygon clipping
    print('\nDownloading images:')
    suffix = '.tif'
    delete_counter = 0 # counter for incorrect S2_RGB
    for satname in im_dict_T1.keys():
        print('\n%s: %d image(s)'%(satname,len(im_dict_T1[satname])))
        
        # create subfolder structure to store the different bands
        filepaths = create_folder_structure(im_folder, satname)
        
        # initialise variables and loop through images
        georef_accs = []; filenames = []; all_names = []; im_epsg = []
        for i in range(len(im_dict_T1[satname])):

            #======================================================================#
            # Metadata: we need first to get info from each image's metadata
            im_meta = im_dict_T1[satname][i]
            
            # get time of acquisition (UNIX time) and convert to datetime
            t = im_meta['properties']['system:time_start']
            im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
            im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')

            # get epsg code
            im_epsg.append(int(im_meta['bands'][0]['crs'][5:]))
            
            # get geometric accuracy
            if satname in ['L5','L7','L8']:
                if 'GEOMETRIC_RMSE_MODEL' in im_meta['properties'].keys():
                    acc_georef = im_meta['properties']['GEOMETRIC_RMSE_MODEL']
                else:
                    acc_georef = 12 # default value of accuracy (RMSE = 12m)
            elif satname in ['S2', 'S2_RGB']:
                # Sentinel-2 products don't provide a georeferencing accuracy (RMSE as in Landsat)
                # but they have a flag indicating if the geometric quality control was passed or failed
                # if passed a value of 1 is stored if failed a value of -1 is stored in the metadata
                skip_geo_check = False
                if 'GEOMETRIC_QUALITY_FLAG' in im_meta['properties'].keys(): 
                    key = 'GEOMETRIC_QUALITY_FLAG'
                elif 'quality_check' in im_meta['properties'].keys():
                    key = 'quality_check'
                else:
                    acc_georef = -1
                    skip_geo_check = True
                if not skip_geo_check:
                    if im_meta['properties'][key] == 'PASSED': acc_georef = 1
                    else: acc_georef = -1 
            georef_accs.append(acc_georef)
            
            # get band information
            bands = dict([])
            im_fn = dict([])
            # delete dimensions key from dict, otherwise the entire image is extracted (don't know why)
            im_bands = im_meta['bands']
            for j in range(len(im_bands)): del im_bands[j]['dimensions']

            # get polygon index if needed (for multiple areas download below)
            if 'polygons_list' in inputs:
                polygon_index = im_meta['polygon_index']

            #======================================================================#
            # Landsat 5 download
            if satname == 'L5':
                bands[''] = [im_bands[0], im_bands[1], im_bands[2], im_bands[3],
                             im_bands[4], im_bands[7]] 
                im_fn[''] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' +\
                            datetime.now().strftime('%H-%M-%S') + suffix 
                # if two images taken at the same date add 'dup' to the name (duplicate)
                if any(im_fn[''] in _ for _ in all_names):
                    im_fn[''] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' +\
                                datetime.now().strftime('%H-%M-%S') + '_dup' + suffix
                all_names.append(im_fn[''])
                filenames.append(im_fn[''])
                # download .tif from GEE
                while True:
                    try:
                        im_ee = ee.Image(im_meta['id'])
                        if 'polygons_list' in inputs:
                            local_data = download_tif(im_ee, inputs['polygons_list'][polygon_index], bands[''], filepaths[1])
                        else:
                            local_data = download_tif(im_ee, inputs['polygon'], bands[''], filepaths[1])
                        break
                    except:
                        continue
                # rename the file as the image is downloaded as 'data.tif'
                try:
                    os.rename(local_data, os.path.join(filepaths[1], im_fn['']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[1], im_fn['']))
                    os.rename(local_data, os.path.join(filepaths[1], im_fn['']))
                # metadata for .txt file
                filename_txt = im_fn[''].replace('.tif','')
                metadict = {'filename':im_fn[''],'acc_georef':georef_accs[i],
                            'epsg':im_epsg[i]}

            #======================================================================#                  
            # Landsat 7 and 8 download                
            elif satname in ['L7', 'L8']:
                if satname == 'L7':
                    bands['pan'] = [im_bands[8]] # panchromatic band
                    bands['ms'] = [im_bands[0], im_bands[1], im_bands[2], im_bands[3],
                                   im_bands[4], im_bands[9]] # multispectral bands
                else:
                    bands['pan'] = [im_bands[7]] # panchromatic band
                    bands['ms'] = [im_bands[1], im_bands[2], im_bands[3], im_bands[4],
                                   im_bands[5], im_bands[11]] # multispectral bands
                for key in bands.keys():
                    im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' +\
                                 key + '_' + datetime.now().strftime('%H-%M-%S') + suffix
                # if two images taken at the same date add 'dup' to the name (duplicate)
                if any(im_fn['pan'] in _ for _ in all_names):
                    for key in bands.keys():
                        im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' +\
                                     key + '_' + datetime.now().strftime('%H-%M-%S') + '_dup' + suffix                    
                all_names.append(im_fn['pan'])
                filenames.append(im_fn['pan'])           
                # download .tif from GEE (panchromatic band and multispectral bands)
                while True:
                    try:
                        im_ee = ee.Image(im_meta['id'])
                        if 'polygons_list' in inputs:
                            local_data_pan = download_tif(im_ee, inputs['polygons_list'][polygon_index], bands['pan'], filepaths[1])
                            local_data_ms = download_tif(im_ee, inputs['polygons_list'][polygon_index], bands['ms'], filepaths[2])
                        else:
                            local_data_pan = download_tif(im_ee, inputs['polygon'], bands['pan'], filepaths[1])
                            local_data_ms = download_tif(im_ee, inputs['polygon'], bands['ms'], filepaths[2])
                        break
                    except:
                        continue
                # rename the files as the image is downloaded as 'data.tif'
                try: # panchromatic
                    os.rename(local_data_pan, os.path.join(filepaths[1], im_fn['pan']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[1], im_fn['pan']))
                    os.rename(local_data_pan, os.path.join(filepaths[1], im_fn['pan'])) 
                try: # multispectral
                    os.rename(local_data_ms, os.path.join(filepaths[2], im_fn['ms']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[2], im_fn['ms']))
                    os.rename(local_data_ms, os.path.join(filepaths[2], im_fn['ms']))   
                # metadata for .txt file
                filename_txt = im_fn['pan'].replace('_pan','').replace('.tif','')
                metadict = {'filename':im_fn['pan'],'acc_georef':georef_accs[i],
                            'epsg':im_epsg[i]}
                 
            #======================================================================#
            # Sentinel-2 Level 1-C download

            # TODO: add RE2 band extraction (im_bands[5], 20m resolution)

            elif satname in ['S2']:
                bands['10m'] = [im_bands[1], im_bands[2], im_bands[3], im_bands[7]] # multispectral bands
                bands['20m'] = [im_bands[11]] # SWIR2 band
                bands['60m'] = [im_bands[15]] # QA band
                for key in bands.keys():
                    im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' +\
                                 key + '_' + datetime.now().strftime('%H-%M-%S') + suffix
                # if two images taken at the same date add 'dup' to the name (duplicate)
                if any(im_fn['10m'] in _ for _ in all_names):
                    for key in bands.keys():
                        im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' +\
                                     key + '_' + datetime.now().strftime('%H-%M-%S') + '_dup' + suffix 
                    # also check for triplicates (only on S2 imagery) and add 'tri' to the name
                    if im_fn['10m'] in all_names:
                        for key in bands.keys():
                            im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' +\
                                         key + '_' + datetime.now().strftime('%H-%M-%S') + '_tri' + suffix    
                all_names.append(im_fn['10m'])
                filenames.append(im_fn['10m']) 
                # download .tif from GEE (multispectral bands at 3 different resolutions)
                while True:
                    try:
                        im_ee = ee.Image(im_meta['id'])
                        if 'polygons_list' in inputs:
                            local_data_10m = download_tif(im_ee, inputs['polygons_list'][polygon_index], bands['10m'], filepaths[1])
                            local_data_20m = download_tif(im_ee, inputs['polygons_list'][polygon_index], bands['20m'], filepaths[2])
                            local_data_60m = download_tif(im_ee, inputs['polygons_list'][polygon_index], bands['60m'], filepaths[3])
                        else:
                            local_data_10m = download_tif(im_ee, inputs['polygon'], bands['10m'], filepaths[1])
                            local_data_20m = download_tif(im_ee, inputs['polygon'], bands['20m'], filepaths[2])
                            local_data_60m = download_tif(im_ee, inputs['polygon'], bands['60m'], filepaths[3])
                        break
                    except:
                        continue
                # rename the files as the image is downloaded as 'data.tif'
                try: # 10m
                    os.rename(local_data_10m, os.path.join(filepaths[1], im_fn['10m']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[1], im_fn['10m']))
                    os.rename(local_data_10m, os.path.join(filepaths[1], im_fn['10m'])) 
                try: # 20m
                    os.rename(local_data_20m, os.path.join(filepaths[2], im_fn['20m']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[2], im_fn['20m']))
                    os.rename(local_data_20m, os.path.join(filepaths[2], im_fn['20m'])) 
                try: # 60m
                    os.rename(local_data_60m, os.path.join(filepaths[3], im_fn['60m']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[3], im_fn['60m']))
                    os.rename(local_data_60m, os.path.join(filepaths[3], im_fn['60m']))
                # metadata for .txt file
                filename_txt = im_fn['10m'].replace('_10m','').replace('.tif','')
                metadict = {'filename':im_fn['10m'],'acc_georef':georef_accs[i],
                            'epsg':im_epsg[i]}

            #======================================================================#
            # Sentinel-2 Level 2-A download

            # Note: a weird thing is that often the RGB Level-2A images have large bands of
            # unicolored pixels (black, white, green, etc.), making them unusable. I added some
            # lines of code to delete these images automatically, but a better thing would be to 
            # download an earlier image until we get a correct image. The problem is that we 
            # download images based on image ids, which are set much earlier (check_available_images)
            # - if we want to re-download incorrect images, we'd need to store the polygon(s)
            # corresponding to these incorrect images, and re-launch a only on the incorrect areas, 
            # a little earler (which is not easy to implement...).

            elif satname in ['S2_RGB']:
                bands['10m'] = [im_bands[15], im_bands[16], im_bands[17]] # True Color RGB bands
                for key in bands.keys():
                    im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' +\
                                 key + '_' + datetime.now().strftime('%H-%M-%S') + suffix
                # if two images taken at the same date add 'dup' to the name (duplicate)
                if any(im_fn['10m'] in _ for _ in all_names):
                    for key in bands.keys():
                        im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' +\
                                     key + '_' + datetime.now().strftime('%H-%M-%S') + '_dup' + suffix 
                    # also check for triplicates (only on S2 imagery) and add 'tri' to the name
                    if im_fn['10m'] in all_names:
                        for key in bands.keys():
                            im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' +\
                                         key + '_' + datetime.now().strftime('%H-%M-%S') + '_tri' + suffix    
                all_names.append(im_fn['10m'])
                filenames.append(im_fn['10m']) 
                # download .tif from GEE
                while True:
                    try:
                        im_ee = ee.Image(im_meta['id'])
                        if 'polygons_list' in inputs:
                            local_data_10m = download_tif(im_ee, inputs['polygons_list'][polygon_index], bands['10m'], filepaths[1])
                        else:
                            local_data_10m = download_tif(im_ee, inputs['polygon'], bands['10m'], filepaths[1])
                        break
                    except:
                        continue
                # delete image and go to next one if it has too many black or white pixels (i.e. area 
                # on the limit of a satellite route, recalibrating errors, duplicates, open sea, etc.)
                # TODO: try to download a valid older image instead of just continuing
                if plt.imread(os.path.join(filepaths[1],'data.tif')).mean() < 50 or\
                   plt.imread(os.path.join(filepaths[1],'data.tif')).mean() > 150:
                    os.remove(os.path.join(filepaths[1], 'data.tif'))
                    print('\r%d%%' %int((i+1)/len(im_dict_T1[satname])*100), end='')
                    delete_counter += 1
                    # incorrect_polygons = 
                    inputs['polygons_list'][polygon_index]
                    continue
                # rename the files as images are downloaded as 'data.tif'
                try: # 10m
                    os.rename(local_data_10m, os.path.join(filepaths[1], im_fn['10m']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[1], im_fn['10m']))
                    os.rename(local_data_10m, os.path.join(filepaths[1], im_fn['10m']))
                # metadata for .txt file
                filename_txt = im_fn['10m'].replace('_10m','').replace('.tif','')
                metadict = {'filename':im_fn['10m'],'acc_georef':georef_accs[i],
                            'epsg':im_epsg[i]}

            # write metadata
            with open(os.path.join(filepaths[0],filename_txt + '.txt'), 'w') as f:
                for key in metadict.keys():
                    f.write('%s\t%s\n'%(key,metadict[key]))  

            # print percentage completion for user
            print('\r%d%%' %int((i+1)/len(im_dict_T1[satname])*100), end='')
    
    # print the nb of incorrect S2_RGB that were deleted
    if delete_counter > 0:
                print('\n\n{} images in the list have not been downloaded'.format(delete_counter) +\
                      ' due to too many missing pixels (area on\nthe edge of a satellite route,' +\
                      ' recalibrating errors, duplicates, open sea, etc.).')

    # once all images have been downloaded, load metadata from .txt files
    metadata = get_metadata(inputs)
          
    # merge overlapping images (if the polygon is at the boundary of an image)
    # and images with the exact same date (i.e. set to False if you download 
    # several chunks from the same S2 tile at the same date)
    if (not 'merge' in inputs) and (not 'polygons_list' in inputs):
        inputs['merge'] = True
    elif 'polygons_list' in inputs:
        inputs['merge'] = False

    if ('S2' in metadata.keys()) and (inputs['merge'] == True):
        try:
            metadata = merge_overlapping_images(metadata,inputs)
        except:
            print('WARNING: there was an error while merging overlapping S2 images.')

    # save metadata dict
    with open(os.path.join(im_folder, inputs['sitename'] + '_metadata' + '.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    return metadata


def create_folder_structure(im_folder, satname):
    """
    Create the structure of subfolders on disk for each satellite mission
    before downloading the images.

    Arguments:
    -----------
    im_folder: str 
        folder where the images are to be downloaded
    satname:
        name of the satellite mission
    
    Returns:
    -----------
    filepaths: list of str
        filepaths of the folders that were created
    """ 
    
    # one folder for the metadata (common to all satellites)
    filepaths = [os.path.join(im_folder, satname, 'meta')]
    # subfolders depending on satellite mission
    if satname == 'L5':
        filepaths.append(os.path.join(im_folder, satname, '30m'))
    elif satname in ['L7','L8']:
        filepaths.append(os.path.join(im_folder, satname, 'pan'))
        filepaths.append(os.path.join(im_folder, satname, 'ms'))
    elif satname in ['S2']: 
        filepaths.append(os.path.join(im_folder, satname, '10m'))
        filepaths.append(os.path.join(im_folder, satname, '20m'))
        filepaths.append(os.path.join(im_folder, satname, '60m'))
    elif satname in ['S2_RGB']: 
        filepaths.append(os.path.join(im_folder, satname, '10m'))
    # create the subfolders if they don't exist already
    for fp in filepaths: 
        if not os.path.exists(fp): os.makedirs(fp)
    
    return filepaths        


def download_tif(image, polygon, bandsId, filepath):
    """
    Downloads a .tif image from the GEE server and stores it in a temp file.

    Arguments:
    -----------
    image: ee.Image
        Image object to be downloaded
    polygon: list
        polygon containing the lon/lat coordinates to be extracted
        longitudes in the first column and latitudes in the second column
    bandsId: list of dict
        list of bands to be downloaded
    filepath: location where the temporary file should be saved
    
    Returns:
    -----------
    Downloads an image in a file named data.tif     
      
    """
           
    # for the old version of ee only
    if int(ee.__version__[-3:]) <= 201:
        url = ee.data.makeDownloadUrl(ee.data.getDownloadId({
            'image': image.serialize(),
            'region': polygon,
            'bands': bandsId,
            'filePerBand': 'false',
            'name': 'data',
            }))
        local_zip, headers = urlretrieve(url)
        with zipfile.ZipFile(local_zip) as local_zipfile:
            return local_zipfile.extract('data.tif', filepath)
    
    # for the newer versions of ee
    else:
        # crop image on the server and create url to download
        url = ee.data.makeDownloadUrl(ee.data.getDownloadId({
            'image': image,
            'region': polygon,
            'bands': bandsId,
            'filePerBand': 'false',
            'name': 'data',
            }))
        # download zipfile with the cropped bands
        local_zip, headers = urlretrieve(url)
        # move zipfile from temp folder to data folder
        dest_file = os.path.join(filepath, 'imagezip')
        shutil.move(local_zip,dest_file)
        # unzip file 
        with zipfile.ZipFile(dest_file) as local_zipfile:
            for fn in local_zipfile.namelist():
                local_zipfile.extract(fn, filepath)
            # filepath + filename to single bands
            fn_tifs = [os.path.join(filepath,_) for _ in local_zipfile.namelist()]
        # stack bands into single .tif
        outds = gdal.BuildVRT(os.path.join(filepath,'stacked.vrt'), fn_tifs, separate=True)
        outds = gdal.Translate(os.path.join(filepath,'data.tif'), outds)
        # delete single-band files
        for fn in fn_tifs: os.remove(fn)
        # delete .vrt file
        os.remove(os.path.join(filepath,'stacked.vrt'))
        # delete zipfile
        os.remove(dest_file)
        # delete data.tif.aux (not sure why this is created)
        if os.path.exists(os.path.join(filepath,'data.tif.aux')):
            os.remove(os.path.join(filepath,'data.tif.aux'))
        # return filepath to stacked file called data.tif
        return os.path.join(filepath,'data.tif')


def remove_cloudy_images(im_list, satname, prc_cloud_cover=75):
    """
    Removes very cloudy images from the GEE collection to be downloaded.
   
    Arguments:
    -----------
    im_list: list 
        list of images in the collection
    satname:
        name of the satellite mission
    prc_cloud_cover: int
        percentage of cloud cover acceptable on the images
    
    Returns:
    -----------
    im_list_upt: list
        updated list of images
    """
    
    # remove very cloudy images from the collection
    if satname in ['L5','L7','L8']:
        cloud_property = 'CLOUD_COVER'
    elif satname in ['S2', 'S2_RGB']:
        cloud_property = 'CLOUDY_PIXEL_PERCENTAGE'
    cloud_cover = [_['properties'][cloud_property] for _ in im_list]
    if np.any([_ > prc_cloud_cover for _ in cloud_cover]):
        idx_delete = np.where([_ > prc_cloud_cover for _ in cloud_cover])[0]
        im_list_upt = [x for k,x in enumerate(im_list) if k not in idx_delete]
    else:
        im_list_upt = im_list
        
    return im_list_upt


def filter_S2_collection(im_list):
    """
    Removes duplicates from the GEE collection of Sentinel-2 images (many duplicates)
    Finds the images that were acquired at the same time but have different utm zones.
    
    Arguments:
    -----------
    im_list: list 
        list of images in the collection
    
    Returns:
    -----------
    im_list_flt: list
        filtered list of images
    """

    # get datetimes
    timestamps = [datetime.fromtimestamp(_['properties']['system:time_start']/1000,
                                         tz=pytz.utc) for _ in im_list]
    # get utm zone projections
    utm_zones = np.array([int(_['bands'][0]['crs'][5:]) for _ in im_list])
    if len(np.unique(utm_zones)) == 1: 
        return im_list
    else:
        utm_zone_selected =  np.max(np.unique(utm_zones))
        # find the images that were acquired at the same time but have different utm zones
        idx_all = np.arange(0,len(im_list),1)
        idx_covered = np.ones(len(im_list)).astype(bool)
        idx_delete = []
        i = 0
        while 1:
            same_time = np.abs([(timestamps[i]-_).total_seconds() for _ in timestamps]) < 60*60*24
            idx_same_time = np.where(same_time)[0]
            same_utm = utm_zones == utm_zone_selected
            # get indices that have the same time (less than 24h apart) but not the same utm zone
            idx_temp = np.where([same_time[j] == True and same_utm[j] == False for j in idx_all])[0]
            idx_keep = idx_same_time[[_ not in idx_temp for _ in idx_same_time]]
            # if more than 2 images with same date and same utm, drop the last ones
            if len(idx_keep) > 2:
               idx_temp = np.append(idx_temp,idx_keep[-(len(idx_keep)-2):])
            for j in idx_temp:
                idx_delete.append(j)
            idx_covered[idx_same_time] = False
            if np.any(idx_covered):
                i = np.where(idx_covered)[0][0]
            else:
                break
        # update the collection by deleting all those images that have same timestamp 
        # and different utm projection
        im_list_flt = [x for k,x in enumerate(im_list) if k not in idx_delete]
    
    #print('\nSentinel-2 duplicates removed - {} images kept for download.'.format(len(im_list_flt)))

    return im_list_flt


def get_metadata(inputs):
    """
    Gets the metadata from previously downloaded images by parsing .txt files located 
    in the \meta subfolders.
            
    Arguments:
    -----------
    inputs: dict with the following fields
        'sitename': str
            name of the site of study
        'filepath': str
            filepath to the directory where the images are downloaded
    
    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system       
    """

    # directory containing the images
    if not 'sitename' in inputs:
        inputs['sitename'] = 'Data-{}'.format(datetime.now().strftime('%Y-%m-%d'))
    filepath = os.path.join(inputs['filepath'], inputs['sitename'])
    # initialize metadata dict
    metadata = dict([])
    # loop through the satellite missions
    for satname in ['L5','L7','L8','S2']:
        # if a folder has been created for the given satellite mission
        if satname in os.listdir(filepath):
            # update the metadata dict
            metadata[satname] = {'filenames':[], 'acc_georef':[], 'epsg':[], 'dates':[]}
            # directory where the metadata .txt files are stored
            filepath_meta = os.path.join(filepath, satname, 'meta')
            # get the list of filenames and sort it chronologically
            filenames_meta = os.listdir(filepath_meta)
            filenames_meta.sort()
            # loop through the .txt files
            for im_meta in filenames_meta:
                # read them and extract the metadata info: filename, georeferencing accuracy
                # epsg code and date
                with open(os.path.join(filepath_meta, im_meta), 'r') as f:
                    filename = f.readline().split('\t')[1].replace('\n','')
                    acc_georef = float(f.readline().split('\t')[1].replace('\n',''))
                    epsg = int(f.readline().split('\t')[1].replace('\n',''))
                date_str = filename[0:19]
                date = pytz.utc.localize(datetime(int(date_str[:4]),int(date_str[5:7]),
                                                  int(date_str[8:10]),int(date_str[11:13]),
                                                  int(date_str[14:16]),int(date_str[17:19])))
                # store the information in the metadata dict
                metadata[satname]['filenames'].append(filename)
                metadata[satname]['acc_georef'].append(acc_georef)
                metadata[satname]['epsg'].append(epsg)
                metadata[satname]['dates'].append(date)
                
    # save a .pkl file containing the metadata dict
    with open(os.path.join(filepath, inputs['sitename'] + '_metadata' + '.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata


###################################################################################################
# MERGE IMAGES
###################################################################################################

def merge_overlapping_images(metadata,inputs):
    """
    Merges simultaneous overlapping images that cover the area of interest (e.g.
    when the area of interest is located at the boundary between 2 images, where 
    there is overlap, this function merges the 2 images so that the AOI is covered
    by only 1 image.

    It also merges images with the exact same date, i.e. if you want to download 
    several chunks from the same S2 tile at the same date, you should specify
    inputs['merge']=False.
            
    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    inputs: dict with the following keys
        'sitename': str
            name of the site of study
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
        'filepath': str
            filepath to the directory where the images are downloaded
        
    Returns:
    -----------
    metadata_updated: dict
        updated metadata     
    """
    
    # only for Sentinel-2 at this stage (not sure if this is needed for Landsat images)    
    sat = 'S2'
    filepath = os.path.join(inputs['filepath'], inputs['sitename'])
    filenames = metadata[sat]['filenames']
    # find the pairs of images that are within 5 minutes of each other
    time_delta = 5*60 # 5 minutes in seconds
    dates = metadata[sat]['dates'].copy()
    pairs = []
    for i,date in enumerate(metadata[sat]['dates']):
        # dummy value so it does not match it again
        dates[i] = pytz.utc.localize(datetime(1,1,1) + timedelta(days=i+1))
        # calculate time difference
        time_diff = np.array([np.abs((date - _).total_seconds()) for _ in dates])
        # find the matching times and add to pairs list
        boolvec = time_diff <= time_delta
        if np.sum(boolvec) == 0:
            continue
        else:
            idx_dup = np.where(boolvec)[0][0]
            pairs.append([i,idx_dup])
    # because they could be triplicates in S2 images, adjust the  for consecutive merges
    for i in range(1,len(pairs)): 
        if pairs[i-1][1] == pairs[i][0]: 
            pairs[i][0] = pairs[i-1][0]
    
    # for each pair of image, create a mask and add no_data into the .tif file (this is 
    # needed before merging .tif files)
    for i,pair in enumerate(pairs):
        fn_im = []
        for index in range(len(pair)): 
            # get filenames of all the files corresponding to the each image in the pair
            fn_im.append([os.path.join(filepath, 'S2', '10m', filenames[pair[index]]),
                  os.path.join(filepath, 'S2', '20m',  filenames[pair[index]].replace('10m','20m')),
                  os.path.join(filepath, 'S2', '60m',  filenames[pair[index]].replace('10m','60m')),
                  os.path.join(filepath, 'S2', 'meta', filenames[pair[index]].replace('_10m','').replace('.tif','.txt'))])
            # read that image
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = \
                                    sat_preprocess.preprocess_single(fn_im[index], sat, False) 
            # im_RGB = sat_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9) 
            
            # in Sentinel2 images close to the edge of the image there are some artefacts, 
            # that are squares with constant pixel intensities. They need to be masked in the 
            # raster (GEOTIFF). It can be done using the image standard deviation, which 
            # indicates values close to 0 for the artefacts.  
            if len(im_ms) > 0:
                # calculate image std for the first 10m band
                im_std = sat_tools.image_std(im_ms[:,:,0],1)
                # convert to binary
                im_binary = np.logical_or(im_std < 1e-6, np.isnan(im_std))
                # dilate to fill the edges (which have high std)
                mask10 = morphology.dilation(im_binary, morphology.square(3))
                # mask all 10m bands
                for k in range(im_ms.shape[2]):
                    im_ms[mask10,k] = np.nan
                # mask the 10m .tif file (add no_data where mask is True)
                sat_tools.mask_raster(fn_im[index][0], mask10)
                # create another mask for the 20m band (SWIR1)
                im_std = sat_tools.image_std(im_extra,1)
                im_binary = np.logical_or(im_std < 1e-6, np.isnan(im_std))
                mask20 = morphology.dilation(im_binary, morphology.square(3))     
                im_extra[mask20] = np.nan
                # mask the 20m .tif file (im_extra)
                sat_tools.mask_raster(fn_im[index][1], mask20) 
                # use the 20m mask to create a mask for the 60m QA band (by resampling)
                mask60 = ndimage.zoom(mask20,zoom=1/3,order=0)
                mask60 = transform.resize(mask60, im_QA.shape, mode='constant', order=0,
                                          preserve_range=True)
                mask60 = mask60.astype(bool)
                # mask the 60m .tif file (im_QA)
                sat_tools.mask_raster(fn_im[index][2], mask60)    
            else:
                continue
            
            # make a figure for quality control
            # fig,ax= plt.subplots(2,2,tight_layout=True)
            # ax[0,0].imshow(im_RGB)
            # ax[0,0].set_title('RGB original')
            # ax[1,0].imshow(mask10)
            # ax[1,0].set_title('Mask 10m')
            # ax[0,1].imshow(mask20)  
            # ax[0,1].set_title('Mask 20m')
            # ax[1,1].imshow(mask60)
            # ax[1,1].set_title('Mask 60 m')
        
        # once all the pairs of .tif files have been masked with no_data, merge them using gdal_merge
        fn_merged = os.path.join(filepath, 'merged.tif')
        
        # merge masked 10m bands and remove duplicate file
        gdal_merge_main(['', '-o', fn_merged, '-n', '0', fn_im[0][0], fn_im[1][0]])
        os.chmod(fn_im[0][0], 0o777)
        os.remove(fn_im[0][0])
        os.chmod(fn_im[1][0], 0o777)
        os.remove(fn_im[1][0])
        os.chmod(fn_merged, 0o777)
        os.rename(fn_merged, fn_im[0][0])
        
        # merge masked 20m band (SWIR band)
        gdal_merge_main(['', '-o', fn_merged, '-n', '0', fn_im[0][1], fn_im[1][1]])
        os.chmod(fn_im[0][1], 0o777)
        os.remove(fn_im[0][1])
        os.chmod(fn_im[1][1], 0o777)
        os.remove(fn_im[1][1])
        os.chmod(fn_merged, 0o777)
        os.rename(fn_merged, fn_im[0][1])
    
        # merge QA band (60m band)
        gdal_merge_main(['', '-o', fn_merged, '-n', '0', fn_im[0][2], fn_im[1][2]])
        os.chmod(fn_im[0][2], 0o777)
        os.remove(fn_im[0][2])
        os.chmod(fn_im[1][2], 0o777)
        os.remove(fn_im[1][2])
        os.chmod(fn_merged, 0o777)
        os.rename(fn_merged, fn_im[0][2])
        
        # remove the metadata .txt file of the duplicate image
        os.chmod(fn_im[1][3], 0o777)
        os.remove(fn_im[1][3])
        
    print('%d overlapping Sentinel-2 images have been merged.' % len(pairs))
    
    # update the metadata dict
    metadata_updated = copy.deepcopy(metadata)
    idx_removed = []
    idx_kept = []
    for pair in pairs: idx_removed.append(pair[1])
    for idx in np.arange(0,len(metadata[sat]['dates'])):
        if not idx in idx_removed: idx_kept.append(idx)
    for key in metadata_updated[sat].keys():
        metadata_updated[sat][key] = [metadata_updated[sat][key][_] for _ in idx_kept]
        
    return metadata_updated  


"""
Sub-module: gdal_merge

Project:  InSAR Peppers
Purpose:  Module to extract data from many rasters into one output.
Author:   Frank Warmerdam, warmerdam@pobox.com

Copyright (c) 2000, Atlantis Scientific Inc. (www.atlsci.com)
Copyright (c) 2009-2011, Even Rouault <even dot rouault at mines-paris dot org>
Changes 29Apr2011, anssi.pekkarinen@fao.org
"""

import math
import sys
import time

from osgeo import gdal

try:
    progress = gdal.TermProgress_nocb
except:
    progress = gdal.TermProgress

verbose = 0
quiet = 0


def raster_copy( s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
                 t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
                 nodata=None ):

    if verbose != 0:
        print('Copy %d,%d,%d,%d to %d,%d,%d,%d.'
              % (s_xoff, s_yoff, s_xsize, s_ysize,
             t_xoff, t_yoff, t_xsize, t_ysize ))

    if nodata is not None:
        return raster_copy_with_nodata(
            s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
            t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
            nodata )

    s_band = s_fh.GetRasterBand( s_band_n )
    m_band = None
    # Works only in binary mode and doesn't take into account
    # intermediate transparency values for compositing.
    if s_band.GetMaskFlags() != gdal.GMF_ALL_VALID:
        m_band = s_band.GetMaskBand()
    elif s_band.GetColorInterpretation() == gdal.GCI_AlphaBand:
        m_band = s_band
    if m_band is not None:
        return raster_copy_with_mask(
            s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
            t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
            m_band )

    s_band = s_fh.GetRasterBand( s_band_n )
    t_band = t_fh.GetRasterBand( t_band_n )

    data = s_band.ReadRaster( s_xoff, s_yoff, s_xsize, s_ysize,
                             t_xsize, t_ysize, t_band.DataType )
    t_band.WriteRaster( t_xoff, t_yoff, t_xsize, t_ysize,
                        data, t_xsize, t_ysize, t_band.DataType )

    return 0


def raster_copy_with_nodata( s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
                             t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
                             nodata ):
    try:
        import numpy as Numeric
    except ImportError:
        import Numeric

    s_band = s_fh.GetRasterBand( s_band_n )
    t_band = t_fh.GetRasterBand( t_band_n )

    data_src = s_band.ReadAsArray( s_xoff, s_yoff, s_xsize, s_ysize,
                                   t_xsize, t_ysize )
    data_dst = t_band.ReadAsArray( t_xoff, t_yoff, t_xsize, t_ysize )

    nodata_test = Numeric.equal(data_src,nodata)
    to_write = Numeric.choose( nodata_test, (data_src, data_dst) )

    t_band.WriteArray( to_write, t_xoff, t_yoff )

    return 0


def raster_copy_with_mask( s_fh, s_xoff, s_yoff, s_xsize, s_ysize, s_band_n,
                           t_fh, t_xoff, t_yoff, t_xsize, t_ysize, t_band_n,
                           m_band ):
    try:
        import numpy as Numeric
    except ImportError:
        import Numeric

    s_band = s_fh.GetRasterBand( s_band_n )
    t_band = t_fh.GetRasterBand( t_band_n )

    data_src = s_band.ReadAsArray( s_xoff, s_yoff, s_xsize, s_ysize,
                                   t_xsize, t_ysize )
    data_mask = m_band.ReadAsArray( s_xoff, s_yoff, s_xsize, s_ysize,
                                    t_xsize, t_ysize )
    data_dst = t_band.ReadAsArray( t_xoff, t_yoff, t_xsize, t_ysize )

    mask_test = Numeric.equal(data_mask, 0)
    to_write = Numeric.choose( mask_test, (data_src, data_dst) )

    t_band.WriteArray( to_write, t_xoff, t_yoff )

    return 0


def names_to_fileinfos( names ):
    """
    Translate a list of GDAL filenames, into file_info objects.

    names -- list of valid GDAL dataset names.

    Returns a list of file_info objects.  There may be less file_info objects
    than names if some of the names could not be opened as GDAL files.
    """

    file_infos = []
    for name in names:
        fi = file_info()
        if fi.init_from_name( name ) == 1:
            file_infos.append( fi )

    return file_infos


class file_info:
    """A class holding information about a GDAL file."""

    def init_from_name(self, filename):
        """
        Initialize file_info from filename

        filename -- Name of file to read.

        Returns 1 on success or 0 if the file can't be opened.
        """
        fh = gdal.Open( filename )
        if fh is None:
            return 0

        self.filename = filename
        self.bands = fh.RasterCount
        self.xsize = fh.RasterXSize
        self.ysize = fh.RasterYSize
        self.band_type = fh.GetRasterBand(1).DataType
        self.projection = fh.GetProjection()
        self.geotransform = fh.GetGeoTransform()
        self.ulx = self.geotransform[0]
        self.uly = self.geotransform[3]
        self.lrx = self.ulx + self.geotransform[1] * self.xsize
        self.lry = self.uly + self.geotransform[5] * self.ysize

        ct = fh.GetRasterBand(1).GetRasterColorTable()
        if ct is not None:
            self.ct = ct.Clone()
        else:
            self.ct = None

        return 1

    def report( self ):
        print('Filename: '+ self.filename)
        print('File Size: %dx%dx%d'
              % (self.xsize, self.ysize, self.bands))
        print('Pixel Size: %f x %f'
              % (self.geotransform[1],self.geotransform[5]))
        print('UL:(%f,%f)   LR:(%f,%f)'
              % (self.ulx,self.uly,self.lrx,self.lry))

    def copy_into( self, t_fh, s_band = 1, t_band = 1, nodata_arg=None ):
        """
        Copy this files image into target file.

        This method will compute the overlap area of the file_info objects
        file, and the target gdal.Dataset object, and copy the image data
        for the common window area.  It is assumed that the files are in
        a compatible projection ... no checking or warping is done.  However,
        if the destination file is a different resolution, or different
        image pixel type, the appropriate resampling and conversions will
        be done (using normal GDAL promotion/demotion rules).

        t_fh -- gdal.Dataset object for the file into which some or all
        of this file may be copied.

        Returns 1 on success (or if nothing needs to be copied), and zero one
        failure.
        """
        t_geotransform = t_fh.GetGeoTransform()
        t_ulx = t_geotransform[0]
        t_uly = t_geotransform[3]
        t_lrx = t_geotransform[0] + t_fh.RasterXSize * t_geotransform[1]
        t_lry = t_geotransform[3] + t_fh.RasterYSize * t_geotransform[5]

        # figure out intersection region
        tgw_ulx = max(t_ulx,self.ulx)
        tgw_lrx = min(t_lrx,self.lrx)
        if t_geotransform[5] < 0:
            tgw_uly = min(t_uly,self.uly)
            tgw_lry = max(t_lry,self.lry)
        else:
            tgw_uly = max(t_uly,self.uly)
            tgw_lry = min(t_lry,self.lry)

        # do they even intersect?
        if tgw_ulx >= tgw_lrx:
            return 1
        if t_geotransform[5] < 0 and tgw_uly <= tgw_lry:
            return 1
        if t_geotransform[5] > 0 and tgw_uly >= tgw_lry:
            return 1

        # compute target window in pixel coordinates.
        tw_xoff = int((tgw_ulx - t_geotransform[0]) / t_geotransform[1] + 0.1)
        tw_yoff = int((tgw_uly - t_geotransform[3]) / t_geotransform[5] + 0.1)
        tw_xsize = int((tgw_lrx - t_geotransform[0])/t_geotransform[1] + 0.5) \
                   - tw_xoff
        tw_ysize = int((tgw_lry - t_geotransform[3])/t_geotransform[5] + 0.5) \
                   - tw_yoff

        if tw_xsize < 1 or tw_ysize < 1:
            return 1

        # Compute source window in pixel coordinates.
        sw_xoff = int((tgw_ulx - self.geotransform[0]) / self.geotransform[1])
        sw_yoff = int((tgw_uly - self.geotransform[3]) / self.geotransform[5])
        sw_xsize = int((tgw_lrx - self.geotransform[0]) \
                       / self.geotransform[1] + 0.5) - sw_xoff
        sw_ysize = int((tgw_lry - self.geotransform[3]) \
                       / self.geotransform[5] + 0.5) - sw_yoff

        if sw_xsize < 1 or sw_ysize < 1:
            return 1

        # Open the source file, and copy the selected region.
        s_fh = gdal.Open( self.filename )

        return raster_copy( s_fh, sw_xoff, sw_yoff, sw_xsize, sw_ysize, s_band,
                            t_fh, tw_xoff, tw_yoff, tw_xsize, tw_ysize, t_band,
                            nodata_arg )


def Usage():
    print('Usage: gdal_merge.py [-o out_filename] [-of out_format] [-co NAME=VALUE]*')
    print('                     [-ps pixelsize_x pixelsize_y] [-tap] [-separate] [-q] [-v] [-pct]')
    print('                     [-ul_lr ulx uly lrx lry] [-init "value [value...]"]')
    print('                     [-n nodata_value] [-a_nodata output_nodata_value]')
    print('                     [-ot datatype] [-createonly] input_files')
    print('                     [--help-general]')
    print('')


# =============================================================================
# Program mainline

def gdal_merge_main( argv=None ):

    global verbose, quiet
    verbose = 0
    quiet = 0
    names = []
    format = 'GTiff'
    out_file = 'out.tif'

    ulx = None
    psize_x = None
    separate = 0
    copy_pct = 0
    nodata = None
    a_nodata = None
    create_options = []
    pre_init = []
    band_type = None
    createonly = 0
    bTargetAlignedPixels = False
    start_time = time.time()

    gdal.AllRegister()
    if argv is None:
        argv = sys.argv
    argv = gdal.GeneralCmdLineProcessor( argv )
    if argv is None:
        sys.exit( 0 )

    # Parse command line arguments.
    i = 1
    while i < len(argv):
        arg = argv[i]

        if arg == '-o':
            i = i + 1
            out_file = argv[i]

        elif arg == '-v':
            verbose = 1

        elif arg == '-q' or arg == '-quiet':
            quiet = 1

        elif arg == '-createonly':
            createonly = 1

        elif arg == '-separate':
            separate = 1

        elif arg == '-seperate':
            separate = 1

        elif arg == '-pct':
            copy_pct = 1

        elif arg == '-ot':
            i = i + 1
            band_type = gdal.GetDataTypeByName( argv[i] )
            if band_type == gdal.GDT_Unknown:
                print('Unknown GDAL data type: %s' % argv[i])
                sys.exit( 1 )

        elif arg == '-init':
            i = i + 1
            str_pre_init = argv[i].split()
            for x in str_pre_init:
                pre_init.append(float(x))

        elif arg == '-n':
            i = i + 1
            nodata = float(argv[i])

        elif arg == '-a_nodata':
            i = i + 1
            a_nodata = float(argv[i])

        elif arg == '-f':
            # for backward compatibility.
            i = i + 1
            format = argv[i]

        elif arg == '-of':
            i = i + 1
            format = argv[i]

        elif arg == '-co':
            i = i + 1
            create_options.append( argv[i] )

        elif arg == '-ps':
            psize_x = float(argv[i+1])
            psize_y = -1 * abs(float(argv[i+2]))
            i = i + 2

        elif arg == '-tap':
            bTargetAlignedPixels = True

        elif arg == '-ul_lr':
            ulx = float(argv[i+1])
            uly = float(argv[i+2])
            lrx = float(argv[i+3])
            lry = float(argv[i+4])
            i = i + 4

        elif arg[:1] == '-':
            print('Unrecognized command option: %s' % arg)
            Usage()
            sys.exit( 1 )

        else:
            names.append(arg)

        i = i + 1

    if len(names) == 0:
        print('No input files selected.')
        Usage()
        sys.exit( 1 )

    Driver = gdal.GetDriverByName(format)
    if Driver is None:
        print('Format driver %s not found, pick a supported driver.' % format)
        sys.exit( 1 )

    DriverMD = Driver.GetMetadata()
    if 'DCAP_CREATE' not in DriverMD:
        print('Format driver %s does not support creation and piecewise writing.' % format,
            '\nPlease select a format that does, such as GTiff (the default) or HFA (Erdas Imagine).' )
        sys.exit( 1 )

    # Collect information on all the source files.
    file_infos = names_to_fileinfos( names )

    if ulx is None:
        ulx = file_infos[0].ulx
        uly = file_infos[0].uly
        lrx = file_infos[0].lrx
        lry = file_infos[0].lry

        for fi in file_infos:
            ulx = min(ulx, fi.ulx)
            uly = max(uly, fi.uly)
            lrx = max(lrx, fi.lrx)
            lry = min(lry, fi.lry)

    if psize_x is None:
        psize_x = file_infos[0].geotransform[1]
        psize_y = file_infos[0].geotransform[5]

    if band_type is None:
        band_type = file_infos[0].band_type

    # Try opening as an existing file.
    gdal.PushErrorHandler( 'CPLQuietErrorHandler' )
    t_fh = gdal.Open( out_file, gdal.GA_Update )
    gdal.PopErrorHandler()

    # Create output file if it does not already exist.
    if t_fh is None:

        if bTargetAlignedPixels:
            ulx = math.floor(ulx / psize_x) * psize_x
            lrx = math.ceil(lrx / psize_x) * psize_x
            lry = math.floor(lry / -psize_y) * -psize_y
            uly = math.ceil(uly / -psize_y) * -psize_y

        geotransform = [ulx, psize_x, 0, uly, 0, psize_y]

        xsize = int((lrx - ulx) / geotransform[1] + 0.5)
        ysize = int((lry - uly) / geotransform[5] + 0.5)


        if separate != 0:
            bands=0

            for fi in file_infos:
                bands=bands + fi.bands
        else:
            bands = file_infos[0].bands


        t_fh = Driver.Create( out_file, xsize, ysize, bands,
                              band_type, create_options )
        if t_fh is None:
            print('Creation failed, terminating gdal_merge.')
            sys.exit( 1 )

        t_fh.SetGeoTransform( geotransform )
        t_fh.SetProjection( file_infos[0].projection )

        if copy_pct:
            t_fh.GetRasterBand(1).SetRasterColorTable(file_infos[0].ct)
    else:
        if separate != 0:
            bands=0
            for fi in file_infos:
                bands=bands + fi.bands
            if t_fh.RasterCount < bands :
                print('Existing output file has less bands than the input files.',
                    'You should delete it before. Terminating gdal_merge.')
                sys.exit( 1 )
        else:
            bands = min(file_infos[0].bands,t_fh.RasterCount)

    # Do we need to set nodata value ?
    if a_nodata is not None:
        for i in range(t_fh.RasterCount):
            t_fh.GetRasterBand(i+1).SetNoDataValue(a_nodata)

    # Do we need to pre-initialize the whole mosaic file to some value?
    if pre_init is not None:
        if t_fh.RasterCount <= len(pre_init):
            for i in range(t_fh.RasterCount):
                t_fh.GetRasterBand(i+1).Fill( pre_init[i] )
        elif len(pre_init) == 1:
            for i in range(t_fh.RasterCount):
                t_fh.GetRasterBand(i+1).Fill( pre_init[0] )

    # Copy data from source files into output file.
    t_band = 1

    if quiet == 0 and verbose == 0:
        progress( 0.0 )
    fi_processed = 0

    for fi in file_infos:
        if createonly != 0:
            continue

        if verbose != 0:
            print("")
            print("Processing file %5d of %5d, %6.3f%% completed in %d minutes."
                  % (fi_processed+1,len(file_infos),
                     fi_processed * 100.0 / len(file_infos),
                     int(round((time.time() - start_time)/60.0)) ))
            fi.report()

        if separate == 0 :
            for band in range(1, bands+1):
                fi.copy_into( t_fh, band, band, nodata )
        else:
            for band in range(1, fi.bands+1):
                fi.copy_into( t_fh, band, t_band, nodata )
                t_band = t_band+1

        fi_processed = fi_processed+1
        if quiet == 0 and verbose == 0:
            progress( fi_processed / float(len(file_infos))  )

    # Force file to be closed.
    t_fh = None

# if __name__ == '__main__':
#     sys.exit(gdal_merge_main())


###################################################################################################
# DOWNLOAD OSM DATA
###################################################################################################

def download_footprints(tag, network_type='all_private', buff=None,
                        place=None, polygon=None, address=None, point=None, radius=1000, 
                        save=False, output_path=os.getcwd()):
    """
    To download OSM data, you can use several tools: the overpass-turbo website, direct Overpass
    URL requests,or other third-party libraries, most notably `osmnx`. The function below is a 
    higher-level function to simplify `osmnx` usage.

    Arguments:
    -----------
    tag: str
        OSM tag key (e.g. 'building','landuse', 'place', etc.) to be downloaded. More specific 
        queries ('landuse=industrial') and multiple queries aren't supported - for complex queries
        please use the overpass-turbo online API (or filter the function output afterwards!)
    network_type: str, optional (default: 'all_private')
        if `tag` is set to 'roads', what type of street network to get - one of 'walk', 'bike',
        'drive', 'drive_service', 'all', or 'all_private' (i.e. including private roads).
    buff: int, optional (default: None)
        if not None, buffer the output geometries with the specified distance (in m)
    place: str
        to get data within some place boundaries. Can be a city, a country, or anything that is 
        geocodable and for which OSM has polygon boundaries. If OSM does not have a polygon for
         this place, you can instead us the `address`, `polygon` or `point` arguments
    polygon: shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        the shape to get data within, with coordinates in epsg:4326
    address: str
        the address for which to get polygons (to be used with `radius` argument) 
    point: tuple
        a lat-lng point, with coordinates in epsg:4326
    radius: int, optional (default: 1000)
        distance (in m) from point or address for which to search for footprints
    save: bool, optional (deault: False)
        save output as a geojson
    output_path: str, optional (default: current working directory)
        folder to save the output in

    Returns:
    -----------
    footprints: geodataframe
        footprints as a geodataframe, or as a multipolygon if `buff` is not 'None'
    """
    
    # download roads
    if tag=='roads':
        if place is not None:
            footprints = osmnx.graph_from_place(place, network_type=network_type)      
        elif polygon is not None:
            footprints = osmnx.graph_from_polygon(polygon, network_type=network_type)
        elif address is not None:
            footprints = osmnx.graph_from_address(address, network_type=network_type, dist=radius)
        elif point is not None:
            footprints = osmnx.graph_from_point(point, network_type=network_type, dist=radius)
        
        footprints = osmnx.graph_to_gdfs(footprints, nodes=False) # get geodataframe with road linestrings  
    
    # download other footprints
    else:
        if place is not None:
            footprints = osmnx.footprints.footprints_from_place(place, footprint_type=tag)
        elif polygon is not None:
            footprints = osmnx.footprints.footprints_from_polygon(polygon, footprint_type=tag)
        elif address is not None:
            footprints = osmnx.footprints.footprints_from_address(address, footprint_type=tag, dist=radius)
        elif point is not None:
            footprints = osmnx.footprints.footprints_from_point(point, footprint_type=tag, dist=radius)
    
    if save:
        #footprints = footprints.to_crs('epsg:4326')
        footprints.to_file(os.path.join(output_path, 'footprints.geojson'), driver='GeoJSON')

    if buff is not None:

        # set up projection transformers
        project = pyproj.Transformer.from_proj(
        pyproj.Proj('epsg:4326'), # source coordinate system
        pyproj.Proj('epsg:2154')) # destination coordinate system

        project_back = pyproj.Transformer.from_proj(
        pyproj.Proj('epsg:2154'), # source coordinate system
        pyproj.Proj('epsg:4326')) # destination coordinate system

        # regroup geometries and compute buffers
        footprints = cascaded_union(list(footprints['geometry'])) # create multigeometry
        footprints = transform(project.transform, footprints) # project roads multilinestring to cartesian epsg:2154
        footprints = footprints.buffer(buff) # compute buffer (results in a multipolygon)
        footprints = transform(project_back.transform, footprints) # project back to epsg:4326

    return footprints

