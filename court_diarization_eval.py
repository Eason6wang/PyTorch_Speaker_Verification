#!/usr/bin/env python
# coding: utf-8
import glob
import librosa
import numpy as np
import os
import torch
from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk
import scipy.stats as stats
from dvector_create import concat_segs,get_STFTs, align_embeddings
from dvector_vis import visualization
import pandas
from tqdm import tqdm
import shutil
import sys
import re
import subprocess
from spectralcluster import SpectralClusterer
import numpy as np
import uisrnn
import matplotlib.pyplot as plt


tmp_dir = './tmp/'
shutil.rmtree(tmp_dir, ignore_errors=True)
os.makedirs(tmp_dir, exist_ok=True)


### Concatenate intervals
def concatenate_intervals(df):
    concat_df = []
    cur_speaker = df.iloc[0]['speaker']
    cur_start = df.iloc[0]['start_time']
    cur_end = df.iloc[0]['stop_time']
    for index, row in df.iterrows():
        if row['speaker'] == cur_speaker:
            cur_end = row['stop_time']
        else:
            concat_df.append({'speaker':cur_speaker, 'start_time':cur_start, 'stop_time':cur_end })
            cur_speaker = row['speaker']
            cur_start = row['start_time']
            cur_end = row['stop_time']
    else:
        concat_df.append({'speaker':cur_speaker, 'start_time':cur_start, 'stop_time':cur_end })
    
    return concat_df

### Generate ffmpeg commands
def generate_ffmpeg_commands(concat_df, mp3_file):
    ffm_command_path = tmp_dir + 'ffm_court.txt'
    ffm_file = open(ffm_command_path,'w')
    for index, row in enumerate(concat_df):
        start = float(row['start_time'])
        duration = float(row['stop_time']) - start 
        if duration < 2: continue # skip it if it is too short
        audio_file = tmp_dir + str(index) + '.wav'
        bashCommand = "ffmpeg -ss " + str(round(start,2)) + " -t " + str(round(duration,2)) + " -i " + mp3_file + ' -y -ar 16000 ' +  audio_file + ' 2> /dev/null'
        ffm_file.write(bashCommand + '\n')
    ffm_file.close()
    return ffm_command_path

### Run ffmpeg in parallel
def run_ffmpeg(ffm_command_path):
    bashCommand = "parallel -j 50 :::: " + ffm_command_path
    process = subprocess.Popen(bashCommand.split())#, stdout=None)#subprocess.PIPE)
    output, error = process.communicate()
    

### Create dvector for each audio

###########MULTIPROCESSING#########
'''
from multiprocessing import Pool
audio_files = sorted(glob.glob(tmp_dir + '*.wav'), key=lambda x:int(os.path.basename(x)[:-4]))

def para_create(audio_file):
    speaker_name = re.sub('[^A-Za-z0-9]+','', concat_df[int(os.path.basename(audio_file)[:-4])]['speaker'])
    try:
        times, segs = VAD_chunk(2, audio_file)
    except:
        print(audio_file + ' is broken')
        return
    if segs == []:
        print('No voice activity detected in ' + audio_file)
        return
    concat_seg = concat_segs(times, segs)
    STFT_frames = get_STFTs(concat_seg)
    if not STFT_frames: 
        print('No STFT frames extracted in ' + audio_file)
        return
    STFT_frames = np.stack(STFT_frames, axis=2)
    STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0))).to(device)
    #STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0)), device=device)

    embeddings = embedder_net(STFT_frames)
    aligned_embeddings = align_embeddings(embeddings.detach().cpu().numpy())
    return aligned_embeddings, speaker_name
def para_create_dvectors():
    NUM_PROCESSES = 4
    pool = Pool(NUM_PROCESSES)
    results = pool.map(para_create, audio_files)
    #for _ in tqdm(pool.imap_unordered(para_create, audio_files), total=len(audio_files)):
    #    pass
    results = [res for res in results if res]
    for res in results:
        # for clustering
        speaker_name = res[1]
        train_sequence.append(res[0])
        for embedding in res[0]:
            train_cluster_id.append(res[1])
    train_sequence = np.concatenate(train_sequence,axis=0)
    train_cluster_id = np.asarray(train_cluster_id)
    np.save('court_test_sequence',train_sequence)
    np.save('court_test_cluster_id',train_cluster_id)
'''
############SINGLE PRO##############
def create_dvectors(concat_df, embedder_net, device):
    test_sequences = []
    test_cluster_ids = []
    audio_files = sorted(glob.glob(tmp_dir + '*.wav'), key=lambda x:int(os.path.basename(x)[:-4]))
    vis_file = open(tmp_dir + "visualization.csv","a+")
    for audio_file in tqdm(audio_files):
        speaker_name = re.sub('[^A-Za-z0-9]+','', concat_df[int(os.path.basename(audio_file)[:-4])]['speaker'])
        try:
            times, segs = VAD_chunk(2, audio_file)
        except:
            print(audio_file + ' is broken')
            continue
        if segs == []:
            print('No voice activity detected in ' + audio_file)
            continue
        concat_seg = concat_segs(times, segs)
        STFT_frames = get_STFTs(concat_seg)
        if not STFT_frames: 
            print('No STFT frames extracted in ' + audio_file)
            continue
        STFT_frames = np.stack(STFT_frames, axis=2)
        STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0))).to(device)
        embeddings = embedder_net(STFT_frames) ### slow
        aligned_embeddings = align_embeddings(embeddings.detach().cpu().numpy())
        test_sequences.append(aligned_embeddings)
        for embedding in aligned_embeddings:
            test_cluster_ids.append(speaker_name)
    test_sequences = np.concatenate(test_sequences,axis=0)
    test_cluster_ids = np.asarray(test_cluster_ids)
    return test_sequences, test_cluster_ids

### Spectral Clustering
def spectral_eval(test_sequences, test_cluster_ids, window_size=1500):
    test_size = window_size
    test_sequences = np.array([np.array(test_sequences[i:i + test_size]) for i in range(0, len(test_sequences), test_size)])
    test_cluster_ids = [list(test_cluster_ids[i:i + test_size]) for i in range(0,len(test_cluster_ids),test_size)]
    accuracy_lst = []
    print("Num of speakers | Num of predicted speakers | Accuracy:")
    
    for sequence, cluster_ids in zip(test_sequences, test_cluster_ids):
        if sys.argv[4] == 'skmeans':
            clusterer = SpectralClusterer(min_clusters=3,max_clusters=20,p_percentile=0.92,gaussian_blur_sigma=1.9,metric='skmeans')
        elif sys.argv[4] == 'spherical':
            clusterer = SpectralClusterer(min_clusters=3,max_clusters=20,p_percentile=0.92,gaussian_blur_sigma=1.9,metric='spherical')
        else:
            clusterer = SpectralClusterer(min_clusters=3,max_clusters=20,p_percentile=0.92,gaussian_blur_sigma=1.9,metric='euclidean')
        labels = clusterer.predict(sequence)
        accuracy = uisrnn.compute_sequence_match_accuracy(list(cluster_ids), list(labels))
        print(str(len(set(cluster_ids))) + "               | " +  str(len(set(labels))) + '                         | ' + str(accuracy))
        accuracy_lst.append(accuracy)
    return np.mean(accuracy_lst)

if __name__ == '__main__':
    
    #argv
    #1: ckpt
    #2: eval_csv
    #3: cuda
    #4: cosine
    ### Initialization
    print('############# ' + os.path.basename(str(sys.argv[1])) + ' #############')
    if sys.argv[3] == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    embedder_net = SpeechEmbedder().to(device)
    embedder_net.load_state_dict(torch.load(sys.argv[1]))
    embedder_net.eval()

    eval_csv = pandas.read_csv(sys.argv[2], delimiter=',')
    
    average_lst = []
    for index, row in eval_csv.iterrows():
        if index > 20: break # 10 tests
        df = pandas.read_csv(row['csv_file'], delimiter=',')
        mp3_file = row['mp3_file']

        id_set = set()
        for index, row in df.iterrows():
            id_set.add(row['speaker'])
        if len(id_set) < 5: continue
        
        concat_df = concatenate_intervals(df)
        ffm_path = generate_ffmpeg_commands(concat_df, mp3_file)
        run_ffmpeg(ffm_path)
        try:
            test_sequences, test_cluster_ids = create_dvectors(concat_df, embedder_net, device)
        except Exception as e:
            print(e)
            print(mp3_file + '!!!')
            continue
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
        print("Speakers in the audio:" + str(set(test_cluster_ids)))
        accuracy = spectral_eval(test_sequences, test_cluster_ids)
        print("Accuracy of " + os.path.basename(mp3_file) + ":" + str(accuracy))
        average_lst.append(accuracy)
        shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)


    print("Average accuracy overall:" + str(np.mean(average_lst)))
    
    
