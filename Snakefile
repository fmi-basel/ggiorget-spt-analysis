import matplotlib.pyplot as plt
from tifffile import imread
import pandas as pd
import numpy as np 
import os
import seaborn as sns 
import sys
import src.utils as ut
import time
from tqdm import tqdm

FILENAME = [ '20230526_Rad21-Halo_NIPBL_1C5_0h_1_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230526_Rad21-Halo_NIPBL_1C5_0h_2_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230526_Rad21-Halo_NIPBL_1C5_0h_3_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230526_Rad21-Halo_NIPBL_1C5_6h_1_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230529_Rad21-Halo_NIPBL_1C5_0h_1_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230529_Rad21-Halo_NIPBL_1C5_6h_1_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230529_Rad21-Halo_NIPBL_1C5_6h_2_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230529_Rad21-Halo_Sororin_2D5_0h_1_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230529_Rad21-Halo_Sororin_2D5_0h_2_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230531_Rad21-Halo_NIPBL_1C5_6h_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230531_Rad21-Halo_sororin_2D5_0h_1_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230531_Rad21-Halo_sororin_2D5_0h_2_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230602_Rad21-Halo_sororin_2D5_3h_1_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230602_Rad21-Halo_sororin_2D5_3h_2_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230602_Rad21-Halo_sororin_2D5_3h_3_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230605_Rad21-Halo_G6_1_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230605_Rad21-Halo_G6_2_FullseqTIRF-Cy5-mCherryGFPWithSMB',
 '20230605_Rad21-Halo_G6_3_FullseqTIRF-Cy5-mCherryGFPWithSMB']

CHANNEL = ['w1','w2']

file_path = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/results/test_tracking_without_gaps/'

df_tracks_corrected = expand(file_path+'{file_name}/tracks_corrected__{file_name}.csv',
                            file_name = FILENAME)

path_to_im = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/images/{file_name}_{channel}.tif'

path_to_im_w2 = '/tungstenfs/scratch/ggiorget/nessim/cohesin_live_cell_analysis/images/{file_name}_w2.tif'

sd = file_path+'{file_name}/sd__{file_name}__{channel}.npy'
df_detection = file_path+'{file_name}/detections__{file_name}__{channel}.csv'

labels_w2 = file_path+'{file_name}/labels__{file_name}__w2.npy'

df_tracks_in_cell = file_path+'{file_name}/tracks_in_cell__{file_name}__{channel}.csv'
df_tracks = file_path+'{file_name}/tracks__{file_name}__{channel}.csv'
df_track_corrected = file_path+'{file_name}/tracks_corrected__{file_name}.csv'

new_labels = expand(file_path+'{file_name}/new_labels__{file_name}__w2.npy',
                    file_name = FILENAME)

new_label = file_path+'{file_name}/new_labels__{file_name}__w2.npy'

df_tracks_in_cell_majority = file_path+'{file_name}/tracks_in_cell_majority__{file_name}__{channel}.csv'

df_tracks_in_cells_majority = expand(file_path+'{file_name}/tracks_in_cell_majority__{file_name}__{channel}.csv',
                                    file_name = FILENAME,
                                    channel = CHANNEL)
# print(df_tracks_in_cells_majority)
dfs_tracks_merged = expand(file_path+'{file_name}/tracks_merged__{file_name}.csv',
                            file_name = FILENAME)

df_tracks_merged = file_path+'{file_name}/tracks_merged__{file_name}.csv'

df_msds_cor = expand(file_path+'{file_name}/msd_corrected__{file_name}.csv',
                            file_name = FILENAME)

df_msd_cor = file_path+'{file_name}/msd_corrected__{file_name}.csv'

df_msds_uncor = expand(file_path+'{file_name}/msd_uncorrected__{file_name}.csv',
                            file_name = FILENAME)

df_msd_uncor = file_path+'{file_name}/msd_uncorrected__{file_name}.csv'

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
    script:
        'src/hmax_snakemake.py'

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
            if i.split('/')[8] == output[0].split('/')[8]: # check that the input and output are named the same to be able to merge the two channels together correctly 
            # should be [7] because it corresponds to the name of the file (wildcard file_name) it's [8] because of the fact that there is no gaps
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
