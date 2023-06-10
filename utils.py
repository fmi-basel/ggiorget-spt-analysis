import numpy as np
import pandas as pd
from skimage.feature import blob_log
import scipy.optimize as opt
from skimage.morphology import disk
from skimage.morphology import extrema
from laptrack import LapTrack
from skimage.measure import regionprops_table
from skimage.transform import resize
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.segmentation import clear_border
from multiprocessing import Pool,get_context
from functools import partial
import secrets
from laptrack.metric_utils import LabelOverlap
from itertools import product
from skimage.measure import regionprops_table
import os

# Fitting
EPS = 1e-4

def gauss_2d(xy:tuple, amplitude, x0, y0, sigma_xy, offset):
    """2D gaussian."""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    gauss = offset + amplitude * np.exp(
        -(
            ((x - x0) ** (2) / (2 * sigma_xy ** (2)))
            + ((y - y0) ** (2) / (2 * sigma_xy ** (2)))
        )
    )
    return gauss

def gauss_single_spot(image: np.ndarray, c_coord: float, r_coord: float, crop_size=4,EPS = 1e-4) -> tuple:
    """Gaussian prediction on a single crop centred on spot."""
    start_dim1 = np.max([int(np.round(r_coord - crop_size // 2)), 0])
    if start_dim1 < len(image) - crop_size:
        end_dim1 = start_dim1 + crop_size
    else:
        start_dim1 = len(image) - crop_size
        end_dim1 = len(image)

    start_dim2 = np.max([int(np.round(c_coord - crop_size // 2)), 0])
    if start_dim2 < len(image) - crop_size:
        end_dim2 = start_dim2 + crop_size
    else:
        start_dim2 = len(image) - crop_size
        end_dim2 = len(image)

    crop = image[start_dim1:end_dim1, start_dim2:end_dim2]

    x = np.arange(0, crop.shape[1], 1)
    y = np.arange(0, crop.shape[0], 1)
    xx, yy = np.meshgrid(x, y)
  
    # Guess intial parameters
    x0 = int(crop.shape[0] // 2)  # Center of gaussian, middle of the crop 
    y0 = int(crop.shape[1] // 2)  # Center of gaussian, middle of the crop 
    sigma = max(*crop.shape) * 0.1  # SD of gaussian, 10% of the crop
    amplitude_max = max(np.max(crop) / 2, np.min(crop))  # Height of gaussian, maximum value
    initial_guess = [amplitude_max, x0, y0, sigma, 0]

    # Parameter search space bounds
    lower = [np.min(crop), 0, 0, 0, -np.inf]
    upper = [
        np.max(crop) + EPS,
        crop_size,
        crop_size,
        np.inf,
        np.inf,
    ]
    bounds = [lower, upper]
    try:
        popt, pcov = opt.curve_fit(
            gauss_2d,
            (xx.ravel(), yy.ravel()),
            crop.ravel(),
            p0=initial_guess,
            bounds=bounds,
        )
        sd = np.sqrt(np.diag(pcov))
    except RuntimeError:
        #print('Runtime')
        return r_coord, c_coord, 0,0

    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1
    sdx = sd[1]
    sdy = sd[2]

    # If predicted spot is out of the border of the image
    if x0 >= image.shape[1] or y0 >= image.shape[0]:
        return r_coord, c_coord, 0,0

    return y0, x0, sdx,sdy

def find_start_end(coord, img_size, crop_size):
    start_dim = np.max([int(np.round(coord - crop_size // 2)), 0])
    if start_dim < img_size - crop_size:
        end_dim = start_dim + crop_size
    else:
        start_dim = img_size - crop_size
        end_dim = img_size

    return start_dim, end_dim

# LoG
def get_loc(im:np.array,frame:int,mins:float,maxs:float,thresh:float,nums:int=10 )-> pd.DataFrame:

    """Function to return localizations from a laptrack detection

    Args:
        im (np.array): input image
        mins (float): minimum sigma used for the detection see skimage.feature blob_log for more details
        maxs (float): maximum sigma used for the detection see skimage.feature blob_log for more details
        nums (float): number of sigma tested for the detection see skimage.feature blob_log for more details
        thresh (float): relative threshold used for the detection see skimage.feature blob_log for more details. Defaults to 10.

    Returns:
        pd.DataFrame: dataframe of all the localizations (gaussian fitted)
    """
    
    ima = im[frame].copy()
    ima = normalize(ima)
    
    df = lap(ima,mins=mins,maxs=maxs,nums=nums,thresh=thresh)
    x_loc =[]
    y_loc =[]
    for i in df.iloc:
        y,x,*_ = gauss_single_spot(ima,i.x,i.y)
        x_loc.append(x)
        y_loc.append(y)

    df['x'] = x_loc
    df['y'] = y_loc

    return df

def lap(im:np.array,mins:float,maxs:float,thresh:float,nums:int=10) -> pd.DataFrame: 
    """Function to compute laptrack spot detection

    Args:
        im (np.array): input image
        mins (float): minimum sigma used for the detection see skimage.feature blob_log for more details
        maxs (float): maximum sigma used for the detection see skimage.feature blob_log for more details
        nums (float): number of sigma tested for the detection see skimage.feature blob_log for more details; Default to
        thresh (float): relative threshold used for the detection see skimage.feature blob_log for more details

    Returns:
        pd.DataFrame: df of all the detections
    """
    images = im

    _spots = blob_log(images.astype(float), min_sigma=mins, max_sigma=maxs, num_sigma=nums,threshold_rel=thresh)
    rad = [np.sqrt(2*_spots[x][-1]) for x in range(len(_spots))]
    df = pd.DataFrame(_spots, columns=["y", "x", "sigma"])
    
    df['radius'] = rad
    
    df = df[df.radius > 1.2]
    
    df.reset_index(drop=True,inplace=True)
    return df

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

def hmax_3D(raw_im: np.ndarray,frame: int,sd: float,n:int = 2,thresh: float = 0.5,threads:int = 10) -> pd.DataFrame:
    """_summary_

    Args:
        raw_im (np.array): the raw image to segment
        frame (int): the frame to segment
        sd (float): the sd of the peak intensity (threshold of segmentation)
        n (int, optional): how much brighter than the sd of the whole image to threshold
        thresh (float, optional): threshold for the gaussian fitting filter. Filter on the standard deviation of the fit (based on the covariance of the parameters) Defaults to 0.5.

    Returns:
        pd.DataFrame: Dataframe of sub-pixel localizations of the detected spots
    """
    #detect the spots
    im_mask = extrema.h_maxima(raw_im[frame],h=n*int(sd))

    # extract the points and fit gaussian
    z,y,x = np.nonzero(im_mask)

    k = [(raw_im[frame,j],x[i],y[i]) for (i,j) in zip(range(len(x)),z)]

    with get_context("fork").Pool(processes=threads) as p:
       x_s,y_s,sdx_fit,sdy_fit = zip(*(p.starmap(gauss_single_spot,k)))

    # create a dataframe with sub pixel localization
    df_loc = pd.DataFrame([x_s,y_s,sdx_fit,sdy_fit]).T
    df_loc.rename(columns={0:'x',1:'y',2:'sd_fit_x',3:'sd_fit_y'},inplace=True)
    df_loc['frame'] = [frame] * len(df_loc)
    df_loc['z'] = z

    # filter the dataframe based on the gaussian fit

    df_loc_filtered = df_loc.query(f'sd_fit_x <{thresh} and sd_fit_y <{thresh}') #remove the bad fit 
    df_loc_filtered = df_loc_filtered[df_loc_filtered.sd_fit_x.values != 0]    #remove the points that were not fitted
    
    df_loc_filtered.rename(columns={'x':'a'},inplace=True)
    df_loc_filtered.rename(columns={'y':'x'},inplace=True)
    df_loc_filtered.rename(columns={'a':'y'},inplace=True)

    return df_loc_filtered

def hmax_detection_fast(raw_im:np.array,frame:int,sd:float,n:int = 2,thresh:float = 0.5) -> pd.DataFrame:
    """_summary_

    Args:
        raw_im (np.array): the raw image to segment
        frame (int): the frame to segment
        sd (float): the sd of the peak intensity (threshold of segmentation)
        n (int, optional): how much brighter than the sd of the whole image to threshold
        thresh (float, optional): threshold for the gaussian fitting filter. Filter on the standard deviation of the fit (based on the covariance of the parameters) Defaults to 0.5.

    Returns:
        pd.DataFrame: Dataframe of sub-pixel localizations of the detected spots
    """
    
    #detect the spots
    im_mask = extrema.h_maxima(raw_im[frame],n*int(sd))

    # extract the points and fit gaussian

    y,x = np.nonzero(im_mask)  # coordinates of every ones

    partial_fit = partial(gauss_single_spot,raw_im[frame])
    
    k = [(x[i],y[i]) for i in range(len(x))]

    with Pool(5) as p:
        x_s,y_s,sdx_fit,sdy_fit = zip(*(p.starmap(partial_fit,k)))
        
    # create a dataframe with sub pixel localization

    df_loc = pd.DataFrame([x_s,y_s,sdx_fit,sdy_fit]).T
    df_loc.rename(columns={0:'x',1:'y',2:'sd_fit_x',3:'sd_fit_y'},inplace=True)
    df_loc['frame'] = [frame] * len(df_loc)
    df_loc
    
    # filter the dataframe based on the gaussian fit

    df_loc_filtered = df_loc.query(f'sd_fit_x <{thresh} and sd_fit_y <{thresh}') #remove the bad fit 
    df_loc_filtered = df_loc_filtered[df_loc_filtered.sd_fit_x.values != 0]    #remove the points that were not fitted
    df_loc_filtered

    return df_loc_filtered

# Stardist

def predict_stardist(im:np.array,size:tuple = (256,256))->np.array:
    """function to predict using stardist

    Args:
        im (np.array): the z_projected image to segment
        size (tuple): the size to expand the image (to be able to have better segmentation). Default: (256,256)

    Returns:
        np.array: the labels with the same shape as the input
    """

    #load model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Resize the image

    images_resized = np.asarray(
                [resize(im[frame, ...], size, anti_aliasing=True) for frame in
                range(im.shape[
                    0])])
    
    # Predict
    os.nice(19)
    labels_resized, _ = zip(*[
            model.predict_instances(normalize(images_resized[frame, ...])) for frame in range(images_resized.shape[0])])
    labels_resized = np.asarray(labels_resized)

    # Resize back the labels

    labels = np.asarray([resize(labels_resized[frame, ...], (im[0].shape[1], im[0].shape[1]), order=0) for frame in range(labels_resized.shape[0])])
    os.nice(0)
    return labels

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

# tracking

def track(df:pd.DataFrame,track_cost_cutoff:float = 3.0,gap_closing_max_frame_count:float = 7.0,gap_closing_cost_cutoff:int = 4,track_dist_metric:str = 'sqeuclidean') ->tuple:
    """Function to track spots in time

    Args:
        df (pd.DataFrame): the detection dataframe for all frames with x,y and frame as columns
        track_cost_cutoff (int, optional): The cutoff distance to consider when tracking (maximum distance to look for a spot to link). Defaults to 2.
        gap_closing_cost_cutoff (int, optional): . Defaults to 2.

    Returns:
        tuple: _description_
    """
    # Track using Lapt track (from emo notebook)
    
    lt = LapTrack(track_cost_cutoff=track_cost_cutoff**2,track_dist_metric=track_dist_metric,
                  gap_closing_cost_cutoff=gap_closing_cost_cutoff**2,gap_closing_max_frame_count=gap_closing_max_frame_count) # track_cost_cutoff and gap_closing_cutoff should be the squared maximum distance", 
    track_df, _, _ = lt.predict_dataframe(df, ["y", "x"], only_coordinate_cols=False,validate_frame=False)
    track_df = track_df.reset_index()
    track_df = track_df[['x','y','frame','track_id']]
    
    return track_df

# rototranslation

def calculate_rototranslation_3D(A, B):
    """Return translation and rotation matrices.
    Args:
        A: coodinates of moving set of points (to which roto translation needs to be applied).
        B: coordinates of fixed set of points.

    Return:
        R, t: rotation and translation matrix
    """
    if None in A or None in B:
        return None, None
    
    A = np.transpose(A)
    B = np.transpose(B)

    num_rows, num_cols = A.shape
    if num_rows != 3 and num_cols !=3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3 and num_cols !=3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def rototranslation_correction_cell(df):
    """Return corrected dataframe given a dataframe with coordinate as input."""
    frames = sorted(df["frame"].unique())
    prev_df = df[df["frame"] == frames[0]].copy()
    #out = prev_df.copy()
    temp_df = prev_df.copy()
    #out[["xres", "yres", "zres"]] = 0
    temp_df[["xres", "yres", "zres"]] = 0
    out_lst = [temp_df]

    rotation = []
    degree_rotation = []
    translation = []
    degree_translation = []
    for idx in np.arange(1, len(frames)):
        curr_df = df[df["frame"] == frames[idx]].copy()
        keep = np.intersect1d(curr_df["track_id"], prev_df["track_id"])
        #print(len(keep))
        #print(keep)
        if len(keep)>3:
            curr4correction = pd.concat([curr_df[curr_df["track_id"] == keptTrack][["x", "y", "z"]] for keptTrack in keep]).values
            prev4correction = pd.concat([prev_df[prev_df["track_id"] == keptTrack][["x", "y", "z"]] for keptTrack in keep]).values

            R, t = calculate_rototranslation_3D(curr4correction, prev4correction)
        else:
            R, t = None, None


        if R is None:
            prev_df = curr_df.copy()
            continue
        # from http://motion.cs.illinois.edu/RoboticSystems/3DRotations.html
        degree_rotation.append(np.arccos((np.trace(R) - 1) / 2))
        degree_translation.append(np.sqrt(np.sum(np.square(t))))
        rotation.append(R)
        translation.append(t)

        prev_df = curr_df.copy()
        curr = np.array(prev_df[["x", "y", "z"]])
        rang = np.arange(len(rotation))
        original_pos = curr.copy()
        for j in rang[::-1]:
            if rotation[j] is not None:
                curr = np.transpose(rotation[j] @ curr.T + translation[j])
        curr_df[["x", "y", "z"]] = curr
        curr_df[["xres", "yres", "zres"]] = curr - original_pos
        #out = pd.concat([out, curr_df])
        out_lst.append(curr_df)
    out = pd.concat(out_lst)
    try:
        out["degree_av_rotation"] = np.mean(degree_rotation)
    except:
        out["degree_av_rotation"] = None
    try:
        out["degree_av_translation"] = np.mean(degree_translation)
    except:
        out["degree_av_translation"] = None
        
    return out

def rototranslation_correction_movie(df):
    #out = pd.DataFrame()
    out_lst = []
    for _, cell_df in df.groupby("label"):
        corrected = rototranslation_correction_cell(cell_df)
        out_lst.append(corrected)
    out = pd.concat(out_lst)

    return out

# MSD

def calculate_single_tamsd(single_traj: pd.DataFrame, min_points: int = 10, radial: bool = False):
    """Calculate trajectory average MSD at all lags.

    Inputs:
        coord: pd.DataFrame containing the coordinates of a given trajectory
        min_points: minimum number of points to calculate the time average MSD
        radial: perform msd on radial distance.
    Return:
        df: pd.DataFrame containing lags and time average MSD"""
    # Calculate pair-wise differences between all timepoints in the trajectory and store it
    # in a matrix
    tvalues = single_traj["frame"].values
    tvalues = tvalues[:, None] - tvalues

    # list of lags
    lags = np.arange(len(single_traj) - min_points) + 1

    final_lags = []
    tamsd = []
    tamsd_count = []
    # Loop over lags
    for lag in lags:
        # find indexes of pairs of timepoints with lag equal to the selected lag
        x, y = np.where(tvalues == lag)

        if len(x) < min_points:
            continue

        if radial:
            sum_nonmean = np.square(
                    single_traj.iloc[x]["distance"].values
                    - single_traj.iloc[y]["distance"].values
                )
            tmp_tamsd = np.mean(sum_nonmean)
            tmp_tamsd_count = len(sum_nonmean)

        else:
            sum_nonmean = np.sum(
                    np.square(
                        single_traj.iloc[x][["x", "y", "z"]].values
                        - single_traj.iloc[y][["x", "y", "z"]].values
                    ),
                    axis=1,
                )
            tmp_tamsd = np.mean(sum_nonmean)
            tmp_tamsd_count = len(sum_nonmean)

        final_lags.append(lag)
        tamsd.append(tmp_tamsd)
        tamsd_count.append(tmp_tamsd_count)

    df = pd.DataFrame({"lags": final_lags, "tamsd": tamsd, "weight":tamsd_count})

    return df

def calculate_all_tamsd(df: pd.DataFrame, min_points: int = 10, min_length: int = 10, radial: bool = False, split_single_traj: bool = False, split_single_traj_val:int = 50):
    """Calculate all time average MSD given a movie and return a DataFrame containing them.

    Inputs:
        traj_file: path to the trajectories file.
        min_points: minimum number of points to consider time average MSD.
        min_length: minimum length of trajectory accepted.
        radial: perform msd on radial distance.

    Return:
        results: pd.DataFrame containing all time average MSD given the trajectories of a movie.
    """


    # output data frame result holder
    #results = pd.DataFrame()
    results_lst = []

    # Loop of tracks
    for track_id in df["track_id"].unique():
        # Extract single trajectory and sort based on time (frame)
        single_traj = df[df["track_id"] == track_id].copy().sort_values(by="frame")
        single_traj.reset_index(drop=True)
        # filter on too short tracks
        if len(single_traj) < min_length:
            continue
        
        if split_single_traj:
            df_tmp = calculate_single_tamsd(
            single_traj.iloc[:split_single_traj_val], min_points=min_points, radial=radial
            )
            df_tmp["uniqueid"] = secrets.token_hex(8)
            df_tmp["track_id"] = track_id
            df_tmp["chunk_of_traj"] = f'first_{split_single_traj_val}'
            results_lst.append(df_tmp)
            
            if len(single_traj) > split_single_traj_val:
                df_tmp = calculate_single_tamsd(
                single_traj.iloc[split_single_traj_val:], min_points=min_points, radial=radial
                )
                df_tmp["uniqueid"] = secrets.token_hex(8)
                df_tmp["track_id"] = track_id
                df_tmp["chunk_of_traj"] = f'second_{split_single_traj_val}'
                df_tmp["trStart"] = single_traj["frame"].min()
                results_lst.append(df_tmp)
        else:
            df_tmp = calculate_single_tamsd(
                single_traj, min_points=min_points, radial=radial
            )
            df_tmp["uniqueid"] = secrets.token_hex(8)
            df_tmp["track_id"] = track_id
            df_tmp["trStart"] = single_traj["frame"].min()
            results_lst.append(df_tmp)
    results = pd.concat(results_lst)

    return results

def mask_tracking_maxoverlab(labels, min_tracklength=0):
    """
    Max-overlap tracking of label images (3D-images (txy)).
    Args:
         labels: (ndarray) Label image series to be tracked
         min_tracklength: (int) Minimum track length to be considered for tracking
    Returns:
         new_labels: (ndarray) label image of tracked cells
         new_track_df: (pandas dataframe) tracking coordinates of tracked cells
    """

    # run modified laptrack, using overlap metrics
    # Calculate the overlap values between every label pair. The paris with no overlap are ignored.
    lo = LabelOverlap(labels)
    overlap_records = []
    for f in range(labels.shape[0] - 1):
        l1s = np.unique(labels[f])
        l1s = l1s[l1s != 0]
        l2s = np.unique(labels[f + 1])
        l2s = l2s[l2s != 0]
        for l1, l2 in product(l1s, l2s):
            overlap, iou, ratio_1, ratio_2 = lo.calc_overlap(f, l1, f + 1, l2)
            overlap_records.append(
                {
                    "frame": f,
                    "label1": l1,
                    "label2": l2,
                    "overlap": overlap,
                    "iou": iou,
                    "ratio_1": ratio_1,
                    "ratio_2": ratio_2,
                }
            )
    overlap_df = pd.DataFrame.from_records(overlap_records)
    overlap_df = overlap_df[overlap_df["overlap"] > 0]
    overlap_df = overlap_df.set_index(["frame", "label1", "label2"]).copy()

    # create coordinate dataframe including labels and centroids
    dfs = []
    for frame in range(len(labels)):
        df = pd.DataFrame(
            regionprops_table(labels[frame], properties=["label", "centroid"])
        )
        df["frame"] = frame
        dfs.append(df)
    coordinate_df = pd.concat(dfs)

    # # LapTrack cannot deal with missing/empty frames, so I include missing frame rows filled with nan here
    frames = pd.Series(range(labels.shape[0]), name='frame')
    coordinate_df = coordinate_df.merge(frames, how='right', on='frame')

    # Define the metric function.
    def metric(c1, c2):
        """
        Metric function: It's an arbitrary function, later used for tracking, that defines the overlab between two
        regions as 'distance'. Uses 1-(label overlap between frame t and t+1 / area of label in frame t+1)
        """
        (frame1, label1), (frame2, label2) = c1, c2
        if frame1 == frame2 + 1:
            tmp = (frame1, label1)
            (frame1, label1) = (frame2, label2)
            (frame2, label2) = tmp
        assert frame1 + 1 == frame2
        ind = (frame1, label1, label2)
        if ind in overlap_df.index:
            ratio_2 = overlap_df.loc[ind]["ratio_2"]
            return 1 - ratio_2
        else:
            return 1

    # execute tracking
    # The defined metric function is used for the frame-to-frame linking (track_dist_metric), gap closing (gap_closing_
    # dist_metric) and the splitting connection (splitting_dist_metric).
    lt = LapTrack(
        track_dist_metric=metric,
        track_cost_cutoff=0.9,
        gap_closing_dist_metric=metric,
        gap_closing_max_frame_count=0,
        splitting_dist_metric=metric,
        splitting_cost_cutoff=0.9,
    )

    track_df, split_df, _ = lt.predict_dataframe(
        coordinate_df, coordinate_cols=["frame", "label"], only_coordinate_cols=False
    )
    track_df = track_df.reset_index()

    # # apply threshold on track length if needed
    # if min_tracklength > 0:
    #     track_df = track_df.groupby('tree_id').filter(lambda x: len(x) >= min_tracklength).reset_index(drop=True)

    # create the new label image time series (tracked)
    new_labels = np.zeros_like(labels)
    for i, row in track_df.iterrows():
        frame = int(row["frame"])
        inds = labels[frame] == row["label"]
        new_labels[frame][inds] = int(row["track_id"]) + 1
    # and a new cleaned-up tracking dataframe that goes with the new label-image
    new_track_df = track_df[["frame", "centroid-0", "centroid-1", "track_id", "tree_id"]].copy()
    new_track_df = new_track_df.rename(
        columns={'centroid-0': 'centroid-y', 'centroid-1': 'centroid-x', 'tree_id': 'parental_id'})
    new_track_df['parental_id'] = new_track_df['parental_id'] + 1
    new_track_df['track_id'] = new_track_df['track_id'] + 1

    return new_labels, new_track_df

def calculate_all_pairwise_tamsd(
    traj_file: str, min_points: int = 10, min_length: int = 10, radial: bool = False):
    """Calculate all time average  pairwise MSD given a movie and return a DataFrame containing them.

    Inputs:
        traj_file: path to the trajectories file.
        min_points: minimum number of points to consider time average MSD.
        min_length: minimum length of trajectory accepted.
        radial: perform msd on radial distance.

    Return:
        results: pd.DataFrame containing all time average pairwise MSD given the trajectories of a movie.
    """

    # Read data from trajectory file
    data = pd.read_csv(traj_file)

    # output data frame result holder
    results = pd.DataFrame()

    for _, df in data.groupby("cell"):
        tracks = df["track"].unique()
        # Loop over tracks
        for i in range(len(tracks)):
            track_id1 = tracks[i]
            for j in range(i + 1, len(tracks)):
                track_id2 = tracks[j]
                # Extract single trajectory and sort based on time (frame)
                first_traj = df[df["track"] == track_id1].copy().sort_values(by="frame")
                second_traj = (
                    df[df["track"] == track_id2].copy().sort_values(by="frame")
                )

                merged = pd.merge(
                    first_traj,
                    second_traj,
                    how="inner",
                    on=["frame", "cell"],
                    suffixes=("_1", "_2"),
                )
                # filter on too short tracks
                if len(merged) < min_length:
                    continue

                merged["distance"] = np.sqrt(
                    np.sum(
                        np.square(
                            merged[["x_1", "y_1", "z_1"]].values
                            - merged[["x_2", "y_2", "z_2"]].values
                        ),
                        axis=1,
                    )
                )

                df_tmp = calculate_single_tamsd(
                    merged, min_points=min_points, radial=radial
                )
                df_tmp["uniqueid"] = secrets.token_hex(8)
                results = pd.concat([results, df_tmp])

    results["traj_file"] = os.path.basename(traj_file)

    return results

def filter_tracks(df):
    n = df.groupby('track_id').size() > 23
    n = n[n == True]
    df_filtered = df[df.track_id.isin(n.index.values)]
    return df_filtered

def calculate_all_pairwise_tamsd(
    data, min_points: int = 10, min_length: int = 10, radial: bool = False
):
    """Calculate all time average  pairwise MSD given a movie and return a DataFrame containing them.

    Inputs:
        traj_file: path to the trajectories file.
        min_points: minimum number of points to consider time average MSD.
        min_length: minimum length of trajectory accepted.
        radial: perform msd on radial distance.

    Return:
        results: pd.DataFrame containing all time average pairwise MSD given the trajectories of a movie.
    """

    # output data frame result holder
    results = pd.DataFrame()

    for _, df in data.groupby("label"):
        tracks = df["track_id"].unique()
        # Loop over tracks
        for i in range(len(tracks)):
            track_id1 = tracks[i]
            for j in range(i + 1, len(tracks)):
                track_id2 = tracks[j]
                # Extract single trajectory and sort based on time (frame)
                first_traj = df[df["track_id"] == track_id1].copy().sort_values(by="frame")
                second_traj = (
                    df[df["track_id"] == track_id2].copy().sort_values(by="frame")
                )

                merged = pd.merge(
                    first_traj,
                    second_traj,
                    how="inner",
                    on=["frame", "label"],
                    suffixes=("_1", "_2"),
                )
                # filter on too short tracks
                if len(merged) < min_length:
                    continue

                merged["distance"] = np.sqrt(
                    np.sum(
                        np.square(
                            merged[["x_1", "y_1", "z_1"]].values
                            - merged[["x_2", "y_2", "z_2"]].values
                        ),
                        axis=1,
                    )
                )

                df_tmp = calculate_single_tamsd(
                    merged, min_points=min_points, radial=radial
                )
                df_tmp["uniqueid"] = secrets.token_hex(8)
                results = pd.concat([results, df_tmp])

    return results