"""
    Processing raw data to csv file and do the synchronization
"""

import os

import pandas as pd
from tqdm import tqdm

FORMAL_DATA_PATH = '/mnt/SART_Paper/Formal_data'
USER_CSV_PATH = '/mnt/SART_Paper/user_eeg_csv'


def parse_data(eeg_text):
    col = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8',
           'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'P7', 'P3',
           'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2', 'HEO', 'VEO']

    eeg = pd.read_csv(eeg_text, sep='\t', index_col=False)
    eeg = eeg[col]

    return eeg


def GetRatingList(EPRIME_txt, trig_list):
    eprime_data = pd.read_csv(EPRIME_txt, sep='\t', skiprows=1, encoding='utf-16')
    size_of_trig = len(trig_list)

    rating_list = [None]*size_of_trig
    rating_RT_list = [None]*size_of_trig
    thought_list = [None]*size_of_trig

    df = eprime_data[['Rating.RESP', 'Rating.RT', 'ThoughtContent.RESP', 'triggerx']]
    df = df.loc[df['triggerx'] == 255].values
    # print(df.shape)

    assert(df.shape[0] == trig_list.count("target"))

    index = 0
    for i in range(size_of_trig):
        if(trig_list[i] == "target"):
            rating_list[i] = df[index, 0]
            rating_RT_list[i] = df[index, 1]
            thought_list[i] = df[index, 2]
            index += 1

    return rating_list, rating_RT_list, thought_list


def merge_trigger(EEG, trigger_csv, EEG_csv, EPRIME_txt=None, main=False):
    normal = 36
    response = 68
    target = 12
    target_res = 20
    trial = 255
    # COL = ['FP1','FP2','F7','F3','FZ','F4','F8','FT7','FC3','FCZ','FC4','FT8','T7','C3','CZ','C4','T8','M2','TP7','CP3','CPZ','CP4','TP8','P7','P3','PZ','P4','P8','O1','OZ','O2','HEO','VEO','trigger']

    trigger = pd.read_csv(trigger_csv, sep="\t")
    trig_type = trigger[['type']].values
    trig_latency = trigger[['latency']].values

    # initiate an empty list
    trig_list = [None]*EEG.shape[0]

    for i in range(trig_type.shape[0]):
        if trig_type[i, 0] == normal:
            trig_list[int(trig_latency[i, 0])] = "normal"
        elif trig_type[i, 0] == response:
            trig_list[int(trig_latency[i, 0])] = "response"
        elif trig_type[i, 0] == target:
            trig_list[int(trig_latency[i, 0])] = "target"
        elif trig_type[i, 0] == target_res:
            trig_list[int(trig_latency[i, 0])] = "target_response"
        elif trig_type[i, 0] == trial:
            trig_list[int(trig_latency[i, 0])] = "probe"
        else:
            print("ERROR OCCURS")

    if main == True:
        Rating_list, Rating_RT_list, thought_list = GetRatingList(EPRIME_txt, trig_list)
        Rating = {"Rating": Rating_list}
        Rating_RT = {"Rating_RT": Rating_RT_list}
        Thought = {"Thought": thought_list}

        # Rating DataFrame
        df_rating = pd.DataFrame(Rating)
        df_rating_RT = pd.DataFrame(Rating_RT)
        df_thought = pd.DataFrame(Thought)

    # EEG dataframe
    trig = {"trigger": trig_list}
    trig_df = pd.DataFrame(trig)
    df = pd.concat([EEG, trig_df], axis=1)

    # Concat
    if main == True:
        df = pd.concat([df, df_rating, df_rating_RT, df_thought], axis=1)

    df.to_csv(EEG_csv)
    # print("Merge to ",EEG_csv)


if __name__ == "__main__":
    file_path_list = [os.path.join(FORMAL_DATA_PATH, fn) for fn in sorted(os.listdir(FORMAL_DATA_PATH))]
    assert len(file_path_list) == 72

    for i, file_path in tqdm(enumerate(file_path_list)):
        data_path = file_path
        try:
            path = data_path.split('/')
            user_name = path[-1]
            user_id = user_name.split('_')[0]

            #print('user id: {}'.format(user_id))

            EEG_format = "txt"
            EPRIME_format = "txt"
            trigger_format = "csv"
            output_format = "csv"

            # pre
            EEG_txt = os.path.join(data_path, user_name+"_eegoutput_prerest."+EEG_format)
            trigger_csv = os.path.join(data_path, user_name+"_eegtrigger_prerest."+trigger_format)
            EEG_csv = os.path.join(USER_CSV_PATH, 'pre', "user"+user_id+"_pre_EEG."+output_format)

            #print("Processing EEG's Pre")
            EEG = parse_data(EEG_txt)
            merge_trigger(EEG, trigger_csv, EEG_csv)

            # post
            EEG_txt = os.path.join(data_path, user_name+"_eegoutput_postrest."+EEG_format)
            trigger_csv = os.path.join(data_path, user_name+"_eegtrigger_postrest."+trigger_format)
            EEG_csv = os.path.join(USER_CSV_PATH, 'post', "user"+user_id+"_post_EEG."+output_format)

            #print("Processing EEG's Post")
            EEG = parse_data(EEG_txt)
            merge_trigger(EEG, trigger_csv, EEG_csv)

            # main
            EEG_txt = os.path.join(data_path, user_name+"_eegoutput_main."+EEG_format)
            trigger_csv = os.path.join(data_path, user_name+"_eegtrigger_main."+trigger_format)
            EEG_csv = os.path.join(USER_CSV_PATH, 'main', "user"+user_id+"_main_EEG."+output_format)
            EPRIME_txt = os.path.join(data_path, user_name+"_eprime_mainoutput."+EPRIME_format)

            #print("Processing EEG's Main")
            EEG = parse_data(EEG_txt)
            merge_trigger(EEG, trigger_csv, EEG_csv, EPRIME_txt, main=True)
        except:
            print(data_path)
