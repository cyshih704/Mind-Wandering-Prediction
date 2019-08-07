
import os

import matplotlib.pyplot as plt
import numpy as np

'''
    The file segment epoch and do the baseline correction
    saved filename according to the epoch location
    read filename according to dirty or clean data
    ln. 16: select signal type 'Clean' or 'Dirty'
    future work, baseline correction
'''
id_list = ['01', '02', '03', '04', '05', '06', '07', '08',
           '09', '10', '12', '14', '15', '16', '17', '18',
           '19', '20', '21', '23', '24', '25', '26', '28',
           '29', '30', '31', '32', '33', '34', '35', '37', 
           '38', '39', '40', '41', '42', '43', '44', '45',
           '46', '47', '48', '49', '50', '51', '52', '53',
           '54', '55', '56']
id_list = ['48']
SAVED_PATH = './'
EEG_TYPE = 'Dirty'
probe = True # probe = True, before probe ; probe = False, before target
ls = 10

def find_target_resp_label(probe_idx, target_resp_idx, i):
    s_idx = 0 if i == 0 else probe_idx[i-1]
    e_idx = probe_idx[i]

    larger = target_resp_idx > s_idx
    lower = target_resp_idx < e_idx

    return 1 - np.sum(larger & lower)


def seg_data(mode, last_sec):
    '''
        trigger: normal (change slide), probe, response (bottom), target (C appear)
        Rating: 1~7
        Rating_RT: response time
        Thought: 1~5
    '''
    #folderpath = os.path.join('/mnt/cys/npy/{}'.format(mode))

    eeg_list, open_eeg_list, close_eeg_list = [], [], []
    eeg_len_list, open_eeg_len_list, close_eeg_len_list = [], [], []
    RT_list = []
    Rating_list = []
    Trigger_list = []
    Thought_list = []
    c_label_list = []
    for _id in id_list:
        if(EEG_TYPE == 'Clean'):
            filepath = os.path.join(SAVED_PATH, 'npy/ica/{}/{}_{}_{}EEG.npy'.format(mode, _id, mode, EEG_TYPE))
        elif(EEG_TYPE == 'Dirty'):
            filepath = os.path.join(SAVED_PATH, 'npy/no_ica/{}/{}_{}_DirtyEEG.npy'.format(mode, _id, mode))

        dic = np.load(filepath, allow_pickle = True)
        print(_id)
        if(mode == 'main'):
            eeg     = dic.item().get('EEG')        
            Thought = dic.item().get('Thought')
            RT      = dic.item().get('RT')
            Rating  = dic.item().get('Rating')
            Trigger = dic.item().get('Trigger')

            probe_idx = np.where(Trigger == 'probe')[0]
            target_idx = np.where(Trigger == 'target')[0]
            target_resp_idx = np.where(Trigger == 'target_response')[0]

            eeg_person = [] # ten second segment
            rating_person = [] # label (1~7)
            thought_person = [] # thought label
            c_label_person = []
            for i in range(len(probe_idx)):
                c_label = find_target_resp_label(probe_idx, target_resp_idx, i)
                if(probe == True):
                    seg_eeg = eeg[:,probe_idx[i] - last_sec * 1000 : probe_idx[i]] # 32 x 10000  10s before probe
                    #seg_eeg1 = eeg1[:,probe_idx[i] - last_sec * 1000 : probe_idx[i]] # 32 x 10000  10s before probe
                    #plt.plot(seg_eeg[4], label = 'w ICA')
                    #plt.plot(seg_eeg1[4], label = 'w/o ICA')
                    #plt.xlabel('sample point')
                    #plt.ylabel('mV')
                    #plt.legend()
                    #plt.show()
                else:
                    seg_eeg = eeg[:,target_idx[i] - last_sec * 1000 : target_idx[i]] # 32 x 10000 10s before C
                #ref_eeg = eeg[:,probe_idx[i] - int((last_sec + 4) * 1000) : probe_idx[i] - last_sec * 1000]
                #baseline = np.mean(ref_eeg, axis = 1)
                #seg_eeg = seg_eeg - np.expand_dims(baseline, axis = 1)

                eeg_person.append(seg_eeg)
                rating_person.append(Rating[target_idx[i]])
                thought_person.append(Thought[target_idx[i]])
                c_label_person.append(c_label)

            eeg_list.append(np.array(eeg_person))
            Rating_list.append(np.array(rating_person))
            Thought_list.append(np.array(thought_person))
            c_label_list.append(np.array(c_label_person))
        elif(mode == 'pre'):
            close_eeg  = dic.item().get('close_eeg')     
            open_eeg  = dic.item().get('open_eeg')     

            close_eeg_list.append(close_eeg)
            close_eeg_len_list.append(close_eeg.shape[1])
   
            open_eeg_list.append(open_eeg)
            open_eeg_len_list.append(open_eeg.shape[1])         

    if(mode == 'main'):
        saved_dic = {}
        saved_dic['eeg'] = np.array(eeg_list)
        saved_dic['rating'] = np.array(Rating_list) 
        saved_dic['thought'] = np.array(Thought_list) 
        saved_dic['target_label'] = np.array(c_label_list) 
        print(np.array(eeg_list).shape)
        print(np.array(Rating_list).shape)  
        print(np.array(Thought_list).shape)  
        print(np.array(c_label_list).shape)

    elif(mode == 'pre'):
        for i in range(len(close_eeg_list)):
            close_eeg_list[i] = close_eeg_list[i][:, -1000*3*60:]
            open_eeg_list[i] = open_eeg_list[i][:, -1000*3*60:]            
            #close_eeg_list[i] = close_eeg_list[i][:, -np.min(close_eeg_len_list):]
            #open_eeg_list[i] = open_eeg_list[i][:, -np.min(open_eeg_len_list):]

        saved_dic = {}
        saved_dic['open_eeg'] = np.array(open_eeg_list)
        saved_dic['close_eeg'] = np.array(close_eeg_list)

        print(np.array(open_eeg_list).shape)
        print(np.array(close_eeg_list).shape)

    if(probe == True):
        if mode == 'main':
            print('save to {}'.format(os.path.join(SAVED_PATH, 'seg_npy/{}/{}s_{}_seg_data'.format(mode, last_sec, EEG_TYPE))))

            np.savez(os.path.join(SAVED_PATH, 'seg_npy/{}/{}s_{}_seg_data'.format(mode, last_sec, EEG_TYPE)),
                eeg = np.array(eeg_list),
                rating = np.array(Rating_list),
                thought = np.array(Thought_list),
                target_label = np.array(c_label_list))  
        elif mode == 'pre':
            np.savez(os.path.join(SAVED_PATH, 'seg_npy/{}/{}_seg_data'.format(mode, EEG_TYPE)), 
                open_eeg = np.array(open_eeg_list),
                close_eeg = np.array(close_eeg_list))  
    else:
        if mode == 'main':
            print('save to {}'.format(os.path.join(SAVED_PATH, 'seg_npy/{}/{}s_{}_seg_data_wC'.format(mode, last_sec, EEG_TYPE))))
            np.savez(os.path.join(SAVED_PATH, 'seg_npy/{}/{}s_{}_seg_data_wC'.format(mode, last_sec, EEG_TYPE)),
                eeg = np.array(eeg_list),
                rating = np.array(Rating_list),
                thought = np.array(Thought_list),
                target_label = np.array(c_label_list))  
        elif mode == 'pre':
            np.savez(os.path.join(SAVED_PATH, 'seg_npy/{}/{}_seg_data_wC'.format(mode, EEG_TYPE)),
                open_eeg = np.array(open_eeg_list),
                close_eeg = np.array(close_eeg_list))  
def make_directory():
    if not os.path.exists(os.path.join(SAVED_PATH, 'seg_npy/main')):
        os.makedirs(os.path.join(SAVED_PATH, 'seg_npy/main'))
    if not os.path.exists(os.path.join(SAVED_PATH, 'seg_npy/pre')):
        os.makedirs(os.path.join(SAVED_PATH, 'seg_npy/pre'))
    if not os.path.exists(os.path.join(SAVED_PATH, 'seg_npy/post')):
        os.makedirs(os.path.join(SAVED_PATH, 'seg_npy/post'))
if __name__ == '__main__':
    make_directory()
    seg_data(mode = 'main', last_sec = ls)
