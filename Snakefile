import matplotlib.pyplot as plt
from tifffile import imread
import pandas as pd
import numpy as np 
import os
import seaborn as sns 
import sys
import utils as ut
import time
from tqdm import tqdm

FILENAME = ['20230605_Rad21-Halo_G6_3_FullseqTIRF-Cy5-mCherryGFPWithSMB']

CHANNEL = ['w1','w2']

df_tracks_corrected = expand('/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/tracks_corrected__{file_name}.csv',
                            file_name = FILENAME)

path_to_im = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/images/{file_name}_{channel}.tif'

path_to_im_w2 = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/images/{file_name}_w2.tif'

sd = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/sd__{file_name}__{channel}.npy'
df_detection = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/detections__{file_name}__{channel}.csv'

labels_w2 = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/labels__{file_name}__w2.npy'

df_tracks_in_cell = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/tracks_in_cell__{file_name}__{channel}.csv'
df_tracks = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/tracks__{file_name}__{channel}.csv'
df_track_corrected = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/tracks_corrected__{file_name}.csv'

new_labels = expand('/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/new_labels__{file_name}__w2.npy',
                    file_name = FILENAME)

new_label = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/new_labels__{file_name}__w2.npy'

df_tracks_in_cell_majority = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/tracks_in_cell_majority__{file_name}__{channel}.csv'

df_tracks_in_cells_majority = expand('/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/tracks_in_cell_majority__{file_name}__{channel}.csv',
                                    file_name = FILENAME,
                                    channel = CHANNEL)
# print(df_tracks_in_cells_majority)
dfs_tracks_merged = expand('/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/tracks_merged__{file_name}.csv',
                            file_name = FILENAME)

df_tracks_merged = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/tracks_merged__{file_name}.csv'

df_msds_cor = expand('/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/msd_corrected__{file_name}.csv',
                            file_name = FILENAME)

df_msd_cor = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/msd_corrected__{file_name}.csv'

df_msds_uncor = expand('/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/msd_uncorrected__{file_name}.csv',
                            file_name = FILENAME)

df_msd_uncor = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking/msd_uncorrected__{file_name}.csv'

rule all:
    input: 
        df_tracks_corrected,
        new_labels,
        dfs_tracks_merged,
        df_tracks_in_cells_majority,
        df_msds_uncor,
        df_msds_cor

        

rule compute_hparam:
    input: path_to_im
    output: sd
    priority:
        19
    run: 
        sd_w1 = []
        im = imread(input[0])
        im = np.max(im,axis=1)
        for frame in tqdm(np.random.randint(0,im.shape[0],size=50)):
            sd_w1.append(ut.compute_h_param(im=im,frame=frame))

        sd_w1 = [x for x in sd_w1 if x > 0]
        np.save(output[0],sd_w1)

rule hmax:
    input: path_to_im,sd
    output: df_detection
    threads: 15
    priority:
        0
    run: 
        im = imread(input[0])
        sd = np.load(input[1])
        df_list = []

        for frame in tqdm(range(np.shape(im)[0])):
            if 'w1' in input[0]:
               df = ut.hmax_3D(im,frame=frame,sd=np.mean(sd),threads=threads)
            else:
                df = ut.hmax_3D(im,frame=frame,sd=np.mean(sd),n=4,thresh=0.20)
            df_list.append(df)
        
        df_detection_w1 = pd.concat(df_list,axis=0)
        df_detection_w1.to_csv(output[0],index=False)

rule compute_labels:
    input: path_to_im_w2
    output: labels_w2
    priority:
        19
    run: 
        im = imread(input[0])
        #predict labels
        if 'w1' in input[0]:
            np.save(output[0],np.zeros_like(im[:,0,:,:]))
        else:
            labels = ut.predict_stardist(np.max(im,axis=1))
        #filter labels
        # _,labels_filtered = ut.filter_cells(labels) # make a column that tells which cell is filtered or not
            labels_filtered = labels
            np.save(output[0],labels_filtered)
    

rule track:
    input: df_detection
    output: df_tracks
    priority:
        19
    run: 
        start = time.time()
        df = pd.read_csv(input[0])
        df_tracks = ut.track(df,track_cost_cutoff=3.0,gap_closing_cost_cutoff=7.0,gap_closing_max_frame_count=0)
        # Find the repeated track_id (matched points)
        # df_tracks['z'] = df.z.values
        # df_tracks['cell_id'] = df.label.values

        u, c = np.unique(df_tracks.track_id.values, return_counts=True)
        dup = u[c > 1]
        track_m = df_tracks[df_tracks.track_id.isin(dup)]
        end = time.time()
        print(f'time to track: {end-start/60} minutes')
        track_m.to_csv(output[0])

rule track_cells:
    input: labels_w2
    output: new_label
    priority:
        19
    run: 
        if 'w1' in input[0]:
            np.save(output[0],np.zeros_like(labels_w2))
        else:
            labels = np.load(input[0])
            new_labels,_ = ut.mask_tracking_maxoverlab(labels)
            np.save(output[0],new_labels)


rule assign_label_to_spot:
    input: df_tracks,new_label
    output: df_tracks_in_cell
    priority:
        19
    run: 
        df_tracks = pd.read_csv(input[0])
        labels = np.load(input[1])
        df_list = []
        for i in tqdm(df_tracks.frame.unique()):
            df_in_cell = ut.assign_label_to_spot(labels,i,df_tracks[df_tracks.frame == i].copy(deep=True))
            df_list.append(df_in_cell)

        df_list = pd.concat(df_list)
        df_list.to_csv(output[0],index=False)

rule majority_rule_cell:
    input: df_tracks_in_cell
    output: df_tracks_in_cell_majority
    priority:
        19
    run: 
        df = pd.read_csv(input[0])
        df = ut.majority_rule(df)
        df.to_csv(output[0],index=False)


rule merge_channel:
    input: df_tracks_in_cells_majority
    output: df_tracks_merged
    priority:
        19
    run: 
        df_list = []
        for i in tqdm(input):
            if i.split('/')[7] == output[0].split('/')[7]:
                df = pd.read_csv(i)
                if 'w1' in i:
                    df['channel'] = ['w1'] * len(df)
                else:
                    df['channel'] = ['w2'] * len(df)

                df['z'] = [0] * len(df)
                df_list.append(df)
            else:
                continue
        df = pd.concat(df_list)
        df.to_csv(output[0],index=False)

rule rototranslation:
    input: df_tracks_merged
    output: df_track_corrected
    priority:
        19
    run: 
        df = pd.read_csv(input[0])
        
        df.loc[df[df.channel == 'w2'].index,'track_id'] = df.loc[df[df.channel == 'w2'].index,'track_id']+ np.max(df.track_id.values) # make sure that the tracks are not overlapping

        df_corrected = ut.rototranslation_correction_movie(df)
        df_corrected.to_csv(output[0])

rule compute_MSD_cor:
    input: df_track_corrected
    output: df_msd_cor
    run: 
        df = pd.read_csv(input[0])
        df = ut.filter_tracks(df)
        df_list = []
        for i in tqdm(df.channel.unique()):
            df_chanel = df[df.channel == i]
            df_chanel = ut.calculate_all_tamsd(df_chanel)
            df_chanel['tamsd'] = df_chanel['tamsd'] * 1.5 # scalling factor for 3D to 2D MSD
            df_chanel['channel'] = [i] * len(df_chanel)
            df_list.append(df_chanel)

        df_msd = pd.concat(df_list)
        df_msd.to_csv(output[0])

rule compute_MSD_uncor:
    input: df_tracks_merged
    output: df_msd_uncor
    run: 
        df = pd.read_csv(input[0])
        df = ut.filter_tracks(df)
        df_list = []
        for i in tqdm(df.channel.unique()):
            df_chanel = df[df.channel == i]
            df_chanel = ut.calculate_all_tamsd(df_chanel)
            df_chanel['tamsd'] = df_chanel['tamsd'] * 1.5 # scalling factor for 3D to 2D MSD
            df_chanel['channel'] = [i] * len(df_chanel)
            df_list.append(df_chanel)

        df_msd = pd.concat(df_list)
        df_msd.to_csv(output[0])