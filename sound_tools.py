from __future__ import unicode_literals
from pydub import AudioSegment
from pydub.playback import play
from os import listdir
import os
import random
import json
import pandas as pd
import numpy as np
import librosa
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from youtubesearchpython import VideosSearch
import youtube_dl
import ffmpeg


# used to convert the noisy file to a wav file
def convert_distort():
    mp3_audio = AudioSegment.from_mp3("noisy.mp3")
    mp3_audio.export("noisy.wav", format="wav")

# used to add the distortion to all the audio files
def add_distortion():
    noisy = AudioSegment.from_wav("noisy.wav") - 20
    wav_path = "audio"
    for clip in listdir(wav_path):
        wav_clip = AudioSegment.from_wav(f"{wav_path}/{clip}")
        overlay = wav_clip.overlay(noisy)
        overlay.export(f"audio_augmented/{clip}", format="wav")

# used to combine all the youtube audio files into one
def cat_all_youtube_audio():
    path = "audio_youtube"
    audios = listdir(path)
    combo = AudioSegment.from_wav("audio_youtube/" + audios[0])
    for aud in audios[1:]:
        wav_clip = AudioSegment.from_wav(f"{path}/{aud}")
        combo += wav_clip
    combo.export("combined_audio.wav", format="wav")


combined = AudioSegment.from_wav("/Users/noahweiss/CS Ariel/Projects/Gender Recognition Project/VoxCeleb_gender/combined_audio.wav") - 20

# used to get a second from the combined audio for the i'th second
def get_sec_from_combined_audio(i: int):
    sec = combined[i * 1000 : (i + 1) * 1000]
    return sec

# used to overlay a second from the combined audio onto all the audio files
def overlay_sec_on_all_audio():
    in_path_males = "data/males"
    in_path_females = "data/females"
    out_path_males = "data/males_aug"
    out_path_females = "data/females_aug"
    audios = listdir(in_path_males)
    for aud_file in audios:
        if aud_file[-4:] == '.wav':
            sound = AudioSegment.from_wav(f"{in_path_males}/{aud_file}")
            length = sound.duration_seconds
            if int(length) == 0:
                overl = sound.overlay(get_sec_from_combined_audio(random.randrange(1184)), position=0)
            else:
                overl = sound.overlay(get_sec_from_combined_audio(random.randrange(1184)), position=random.randrange(int(length) * 1000))
            overl.export(f"{out_path_males}/{aud_file[:-3]}wav", format="wav")
    audios = listdir(in_path_females)
    for aud_file in audios:
        if aud_file[-4:] == '.wav':
            sound = AudioSegment.from_wav(f"{in_path_females}/{aud_file}")
            length = sound.duration_seconds
            if int(length) == 0:
                overl = sound.overlay(get_sec_from_combined_audio(random.randrange(1184)), position=0)
            else:
                overl = sound.overlay(get_sec_from_combined_audio(random.randrange(1184)), position=random.randrange(int(length) * 1000))
            overl.export(f"{out_path_females}/{aud_file[:-3]}wav", format="wav")


# used to get the total audio time of the dataset
def total_audio_time():
    in_path = "data/males"
    audios = listdir(in_path)
    sum = 0
    count = 0
    for aud_file in audios:
        if aud_file[-3:] == 'wav':
            count += 1
            sound = AudioSegment.from_wav(f"{in_path}/{aud_file}")
            sum += sound.duration_seconds
    in_path = "data/females"
    audios = listdir(in_path)
    for aud_file in audios:
        if aud_file[-3:] == 'wav':
            count += 1
            sound = AudioSegment.from_wav(f"{in_path}/{aud_file}")
            sum += sound.duration_seconds
    print(sum)
    print(count)
    print("avg:", sum / count)


def exportfile(newAudio, time1, time2, filename, i):
    # Exports to a wav file in the current path.
    newAudio2 = newAudio[time1:time2]
    g = os.listdir()
    if filename[0:-4] + '_' + str(i) + '.wav' in g:
        filename2 = str(i) + '_segment' + '.wav'
        print('making %s' % (filename2))
        newAudio2.export(filename2, format="wav")
    else:
        filename2 = str(i) + '.wav'
        print('making %s' % (filename2))
        newAudio2.export(filename2, format="wav")

    return filename2

def audio_time_features(filename):
    # recommend >0.50 seconds for timesplit
    timesplit = 0.50
    hop_length = 512
    n_fft = 2048

    y, sr = librosa.load(filename)
    duration = float(librosa.core.get_duration(y))

    # Now splice an audio signal into individual elements of 100 ms and extract
    # all these features per 100 ms
    segnum = round(duration / timesplit)
    deltat = duration / segnum
    timesegment = list()
    time = 0

    for i in range(segnum):
        # milliseconds
        timesegment.append(time)
        time = time + deltat * 1000

    newAudio = AudioSegment.from_wav(filename)
    filelist = list()

    for i in range(len(timesegment) - 1):
        filename = exportfile(newAudio, timesegment[i], timesegment[i + 1], filename, i)
        filelist.append(filename)

    featureslist = np.array([0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0])

    # save 100 ms segments in current folder (delete them after)
    for j in range(len(filelist)):
        try:
            features = featurize(filelist[i])
            featureslist = featureslist + features
            os.remove(filelist[j])
        except:
            print('error splicing')
            featureslist.append('silence')
            os.remove(filelist[j])

    # now scale the featureslist array by the length to get mean in each category
    featureslist = featureslist / segnum

    return featureslist

def featurize(wavfile):
    #initialize features 
    hop_length = 512
    n_fft=2048
    #load file 
    y, sr = librosa.load(wavfile)
    #extract mfcc coefficients 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc) 
    #extract mean, standard deviation, min, and max value in mfcc frame, do this across all mfccs
    mfcc_features=np.array([np.mean(mfcc[0]),np.std(mfcc[0]),np.amin(mfcc[0]),np.amax(mfcc[0]),
                            np.mean(mfcc[1]),np.std(mfcc[1]),np.amin(mfcc[1]),np.amax(mfcc[1]),
                            np.mean(mfcc[2]),np.std(mfcc[2]),np.amin(mfcc[2]),np.amax(mfcc[2]),
                            np.mean(mfcc[3]),np.std(mfcc[3]),np.amin(mfcc[3]),np.amax(mfcc[3]),
                            np.mean(mfcc[4]),np.std(mfcc[4]),np.amin(mfcc[4]),np.amax(mfcc[4]),
                            np.mean(mfcc[5]),np.std(mfcc[5]),np.amin(mfcc[5]),np.amax(mfcc[5]),
                            np.mean(mfcc[6]),np.std(mfcc[6]),np.amin(mfcc[6]),np.amax(mfcc[6]),
                            np.mean(mfcc[7]),np.std(mfcc[7]),np.amin(mfcc[7]),np.amax(mfcc[7]),
                            np.mean(mfcc[8]),np.std(mfcc[8]),np.amin(mfcc[8]),np.amax(mfcc[8]),
                            np.mean(mfcc[9]),np.std(mfcc[9]),np.amin(mfcc[9]),np.amax(mfcc[9]),
                            np.mean(mfcc[10]),np.std(mfcc[10]),np.amin(mfcc[10]),np.amax(mfcc[10]),
                            np.mean(mfcc[11]),np.std(mfcc[11]),np.amin(mfcc[11]),np.amax(mfcc[11]),
                            np.mean(mfcc[12]),np.std(mfcc[12]),np.amin(mfcc[12]),np.amax(mfcc[12]),
                            np.mean(mfcc_delta[0]),np.std(mfcc_delta[0]),np.amin(mfcc_delta[0]),np.amax(mfcc_delta[0]),
                            np.mean(mfcc_delta[1]),np.std(mfcc_delta[1]),np.amin(mfcc_delta[1]),np.amax(mfcc_delta[1]),
                            np.mean(mfcc_delta[2]),np.std(mfcc_delta[2]),np.amin(mfcc_delta[2]),np.amax(mfcc_delta[2]),
                            np.mean(mfcc_delta[3]),np.std(mfcc_delta[3]),np.amin(mfcc_delta[3]),np.amax(mfcc_delta[3]),
                            np.mean(mfcc_delta[4]),np.std(mfcc_delta[4]),np.amin(mfcc_delta[4]),np.amax(mfcc_delta[4]),
                            np.mean(mfcc_delta[5]),np.std(mfcc_delta[5]),np.amin(mfcc_delta[5]),np.amax(mfcc_delta[5]),
                            np.mean(mfcc_delta[6]),np.std(mfcc_delta[6]),np.amin(mfcc_delta[6]),np.amax(mfcc_delta[6]),
                            np.mean(mfcc_delta[7]),np.std(mfcc_delta[7]),np.amin(mfcc_delta[7]),np.amax(mfcc_delta[7]),
                            np.mean(mfcc_delta[8]),np.std(mfcc_delta[8]),np.amin(mfcc_delta[8]),np.amax(mfcc_delta[8]),
                            np.mean(mfcc_delta[9]),np.std(mfcc_delta[9]),np.amin(mfcc_delta[9]),np.amax(mfcc_delta[9]),
                            np.mean(mfcc_delta[10]),np.std(mfcc_delta[10]),np.amin(mfcc_delta[10]),np.amax(mfcc_delta[10]),
                            np.mean(mfcc_delta[11]),np.std(mfcc_delta[11]),np.amin(mfcc_delta[11]),np.amax(mfcc_delta[11]),
                            np.mean(mfcc_delta[12]),np.std(mfcc_delta[12]),np.amin(mfcc_delta[12]),np.amax(mfcc_delta[12])])
    
    return mfcc_features

# creates dataset of 720 features for each audio file
def create_720_feature_dataset():
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    MALES_PATH = "data/males"
    FEMALES_PATH = "data/females"
    male_files = listdir(MALES_PATH)
    female_files = listdir(FEMALES_PATH)
    min_amount = min(len(male_files), len(female_files))
    boys = []
    girls = []
    count = 0
    for file in male_files:
        if file[-3:] == 'wav':
            if count >= min_amount:
                break
            features = np.append(featurize(f"{MALES_PATH}/{file}"),audio_time_features(f"{MALES_PATH}/{file}"))
            signal, fs =torchaudio.load(f"{MALES_PATH}/{file}")
            embeddings = classifier.encode_batch(signal)
            embeddings = embeddings.detach().cpu().numpy()
            embedding = embeddings[0][0]
            boys.append(features.tolist() + embedding.tolist())
            # sound = AudioSegment.from_wav(f"{MALES_PATH}/{file}")
            # sound.export(f"{MALES_OUT_PATH}/{file}", format='wav')
            count += 1
    
    count = 0
    for file in female_files:
        if file[-3:] == 'wav':
            if count >= min_amount:
                break
            features = featurize(f"{FEMALES_PATH}/{file}")
            signal, fs =torchaudio.load(f"{FEMALES_PATH}/{file}")
            embeddings = classifier.encode_batch(signal)
            embeddings = embeddings.detach().cpu().numpy()
            embedding = embeddings[0][0]
            girls.append(features.tolist() + embedding.tolist())
            # sound = AudioSegment.from_wav(f"{FEMALES_PATH}/{file}")
            # sound.export(f"{FEMALES_OUT_PATH}/{file}", format='wav')
            count += 1

    print("boys: ", len(boys))
    print("girls: ", len(girls))

    json_obj = {"males": boys, "females": girls}
    with open('data/males_females_audio.json', 'w') as outfile:
        json.dump(json_obj, outfile)


# converts m4a files to wav files
def convert_to_wav():
    MALES_PATH = "/Users/noahweiss/CS Ariel/Projects/Gender Recognition Project/Gender-Voice-Recognition-Project/data/males_m4a"
    FEMALES_PATH = "/Users/noahweiss/CS Ariel/Projects/Gender Recognition Project/Gender-Voice-Recognition-Project/data/females_m4a"
    MALES_OUT_PATH = "/Users/noahweiss/CS Ariel/Projects/Gender Recognition Project/Gender-Voice-Recognition-Project/data/males"
    FEMALES_OUT_PATH = "/Users/noahweiss/CS Ariel/Projects/Gender Recognition Project/Gender-Voice-Recognition-Project/data/females"
    male_files = listdir(MALES_PATH)
    female_files = listdir(FEMALES_PATH)
    for wavfile in male_files:
        if wavfile[-3:] != 'm4a':
            continue
        sound = AudioSegment.from_file(f"{MALES_PATH}/{wavfile}", format='m4a')
        sound.export(f"{MALES_OUT_PATH}/{wavfile[:-4]}.wav", format='wav')
    for wavfile in female_files:
        if wavfile[-3:] != 'm4a':
            continue
        sound = AudioSegment.from_file(f"{FEMALES_PATH}/{wavfile}", format='m4a')
        sound.export(f"{FEMALES_OUT_PATH}/{wavfile[:-4]}.wav", format='wav')


def random_word_pick():
    word_list = []
    w_list = []
    with open("words_alpha.txt", "r") as file:
        for word in file:
            if len(word) >= 5:
                word = word[:-1]
                word_list.append(word)
    w_list = random.sample(word_list, 50)
    return w_list


def get_vid_links():
    vid_links = []
    all_words = []
    with open("words_alpha.txt", "r") as file:
        for word in file:
            all_words.append(word)
    random.shuffle(all_words)
    for word in all_words:
        if len(word) >= 5:
            word = word[:-1]
            search_results = VideosSearch(word, limit=100).result()["result"]
            for i in range(len(search_results)):
                if int(search_results[i]["duration"].split(":")[0]) < 1:
                    vid_links.append(search_results[i]["link"])
                    break
        if len(vid_links) == 50:
            return vid_links


def download_audio(vid_links: list):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio_youtube/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        for vid in vid_links:
            ydl.download([vid])

if __name__ == '__main__':
    # convert_distort()
    # add_distortion()
    # cat_all_youtube_audio()
    # overlay_sec_on_all_audio()
    # take_gender_audios()
    # create_720_feature_dataset()
    # convert_to_wav()
    # total_audio_time()
    # make_new_validated_file()
    # random_word_pick()
    # vl = get_vid_links()
    # download_audio(vl)
    pass
