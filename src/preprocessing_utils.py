import numpy as np
import pandas as pd
from skimage.feature import blob_log
import scipy.optimize as opt
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border
from multiprocessing import Pool,get_context
from functools import partial
import secrets
# H_max 

def heatmap_detection(raw_im:np.array,frame:int,df:pd.DataFrame,name:str)-> tuple:
    """Create a heatmap to compute the threshold for h-max detection

    Args:
        raw_im (np.array): the raw image to segment
        frame (int): the frame to segment
        df (pd.DataFrame): detections (x,y) on the raw image to be able to compute the intensity profiles of the detected spots
        name (str): either 'med' or 'sd'. Whether you want to display the median pixel intensity value around the spots or the sd. In any case it returns the values of the 2

    Returns:
        tuple: heatmap: a n-dimentional array (shape of the image) with either the median pixel intensity value for each bins or the sd of each bin, the median pixel intensity value for all detected spots provided, the sd of the intensity of every provided spots
    """
    # create image with extended boarders to be able to take bbox

    im = np.pad(raw_im[frame],3)

    # create the same for the mask

    im_mask = np.pad(np.ones_like(raw_im[frame],dtype=int),3)

    spot = []
    for i in df.iloc:
        x,y = int(i.x+2),int(i.y+2)
        patch = im[x-2:x+3,y-2:y+3]
        patch_mask = im_mask[x-2:x+3,y-2:y+3]*disk(2)  # get only a disk in the bbox
        spot.append(patch[patch_mask].ravel()) # get a 1d list of all bbox 

    med = np.median(spot,axis=1) # median of bbox where there are spots
    sd = np.std(spot,axis=1) # sd of bbox where there are spots

    df['med'] = med
    df['sd'] = sd

    df_heat = df[['x','y','med','sd']].copy(deep=True)

    x = []
    y = []
    for i,j in zip(df['x'].values,df['y'].values):
        try:
            x.append(i//32)
            y.append(j//32)
        except RuntimeWarning:
            x.append(0)
            y.append(0)

    df_heat['x'] = x#df['x'].values//32
    df_heat['y'] = y#df_heat['y'].values//32 # bin the image to categories to be able to see better the spots 

    heatmap = []
    for i in range(int(max(df_heat.y.values))):
        list_heat_row =[]
        for j in range(int(max(df_heat.x.values))):
            try:
                list_heat_row.append(np.mean(df_heat[(df_heat.x == j) & (df_heat.y == i)][name].values)) # average the region binned (row by row)
            except RuntimeWarning:
                list_heat_row.append(0)
                
        heatmap.append(list_heat_row)

    heatmap = np.array(heatmap)

    return heatmap,sd,med

def compute_h_param(im:np.array,frame:int,mins:float = 1.974 ,maxs:float = 3.0 ,thresh:float = 0.884) -> float:

    # Compute LoG with very high threhsold 

    df = get_loc(im,frame,mins,maxs,thresh)

    # if there is no spot be a bit more gentle with the parameters
    if len(df) == 0:
        return 0
    else:
        #compute the sd of the detected spots

        _,sd,_ = heatmap_detection(im,frame=frame,df=df,name='sd')

        # compute the mean sd across the image 

        mean_sd = np.mean(sd)

        return mean_sd

def manual_h_param(path:str,im:np.array,frame:int)->float:
    """Function to compute the h param from manually picked points on the image

    Args:
        path (str): path where to find the csv docuemtn containing X , Y **in caps** (default in fiji when you export measurements)
        im (np.array): the image where you picked the spots (or crop of the image)
        frame (int): the frame number (careful as python starts at 0 and fiji 1)

    Returns:
        float: the average standard deviation of the intensity around the selected spots to compute hmax 
    """
    # open the csv 
    df = pd.read_csv(path)

    x_loc =[]
    y_loc =[]
    for i in df.iloc:
        y,x,*_ = gauss_single_spot(im[frame],i.X,i.Y)
        x_loc.append(x)
        y_loc.append(y)

    df['x'] = x_loc
    df['y'] = y_loc

    #compute the sd of the detected spots

    _,sd,_ = heatmap_detection(im,frame=frame,df=df,name='sd')

    # compute the mean sd across the image 

    mean_sd = np.mean(sd)

    return mean_sd


# cell assignment 

def filter_cells(labels:np.array,thresh_eccentricity:float = 0.8)-> tuple:
    """Function to filter labeled cells

    Args:
        labels (np.array): labeled image
        thresh_eccentricity (float, optional): the threshold on the eccentricity to filter cells (the close they are to 1 the less round they are ). Defaults to 0.8.

    Returns:
        tuple: the coordinates of the filltered cells and the corresponding labeled images
    """

    #clear the cells that touch border

    labels_cleared = np.asarray([clear_border(labels[frame, ...]) for frame in range(labels.shape[0])]) #labels not touching the border

    #compute region props of the labels computed above 

    dfs = []
    for frame in range(len(labels_cleared)):
        df = pd.DataFrame(
            regionprops_table(labels_cleared[frame], properties=["label", "centroid",'area','bbox','eccentricity'])
        )
        df["frame"] = frame
        dfs.append(df)
    coordinate_df_cleared = pd.concat(dfs)

    # filter based on eccentricity (and size ?)
    coordinate_df_cleared = coordinate_df_cleared.query(f'eccentricity < {thresh_eccentricity}') #filter the eccentricity

    return coordinate_df_cleared,labels_cleared

def assign_label_to_spot(labels:np.array,frame:int,df:pd.DataFrame) -> tuple:

    lab_so = [labels[frame,int(i.y),int(i.x)] for i in df.iloc]
    df['label'] = lab_so
    # df_in_cell = df[df.label != 0] 

    return df

def majority_rule(df:pd.DataFrame):
    for i in df.track_id.unique():
        df.loc[df[df.track_id == i].index,'label'] = max(df[df.track_id == i].label.values,key=list(df[df.track_id == i].label.values).count)
    
    df = df[df.label != 0]
    return df
