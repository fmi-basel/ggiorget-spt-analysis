import numpy as np
import pandas as pd
import scipy.optimize as opt
from multiprocessing import Pool,get_context
from functools import partial
import secrets
from itertools import product
import os

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

def filter_tracks(df, min_length = 23):
    n = df.groupby('track_id').size() > min_length
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
        if len(single_traj) <= min_length:
            continue
        
        if split_single_traj:
            df_tmp = calculate_single_tamsd(
            single_traj.iloc[:split_single_traj_val], min_points=min_points, radial=radial
            )
            df_tmp["uniqueid"] = secrets.token_hex(10)
            df_tmp["track_id"] = track_id
            df_tmp["chunk_of_traj"] = f'first_{split_single_traj_val}'
            results_lst.append(df_tmp)
            
            if len(single_traj) > split_single_traj_val:
                df_tmp = calculate_single_tamsd(
                single_traj.iloc[split_single_traj_val:], min_points=min_points, radial=radial
                )
                df_tmp["uniqueid"] = secrets.token_hex(10)
                df_tmp["track_id"] = track_id
                df_tmp["chunk_of_traj"] = f'second_{split_single_traj_val}'
                df_tmp["trStart"] = single_traj["frame"].min()
                results_lst.append(df_tmp)
        else:
            df_tmp = calculate_single_tamsd(
                single_traj, min_points=min_points, radial=radial
            )
            df_tmp["uniqueid"] = secrets.token_hex(10)
            df_tmp["track_id"] = track_id
            df_tmp["trStart"] = single_traj["frame"].min()
            results_lst.append(df_tmp)
    if len(results_lst) > 0:
        results = pd.concat(results_lst)
    else:
        results = pd.DataFrame(columns=["lags", "tamsd", "weight", "uniqueid", "track_id", "trStart"])

    return results