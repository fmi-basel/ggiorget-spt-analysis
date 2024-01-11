import numpy as np
import pandas as pd
from laptrack import LapTrack
from skimage.measure import regionprops_table
from laptrack.metric_utils import LabelOverlap


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
