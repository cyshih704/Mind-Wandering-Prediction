import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import butter, lfilter, find_peaks
import pandas as pd
import os
import time 
import mne

'''
	The file do the bandpass filer (0.1 ~ 45) and ICA (filter 1 Hz first) 
	if use ICA, save filename as Clean
	if not use ica, save filename as Dirty
	future work, not filter before ICA
'''
# Dirty: no use ICA, Clean: use ICA
EEG_TYPE = 'Clean'
SAVED_PATH = './'

# Channel list with M2 channel
channel_list_M2 = np.array(['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
                         'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T7', 
                         'C3', 'CZ', 'C4', 'T8', 'M2', 'TP7', 'CP3', 
                         'CPZ', 'CP4', 'TP8', 'P7', 'P3', 'PZ', 'P4', 
                         'P8', 'O1', 'OZ', 'O2', 'HEO', 'VEO'])
# Channel list without M2 channel
channel_list = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
                'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T7', 
                'C3', 'CZ', 'C4', 'T8', 'TP7', 'CP3', 
                'CPZ', 'CP4', 'TP8', 'P7', 'P3', 'PZ', 'P4', 
                'P8', 'O1', 'OZ', 'O2', 'HEO', 'VEO']

class csv_to_npy:
    def __init__(self):
        self.csv_folder = '/mnt/cys/EEG_data_rating'

    def csv2numpy(self, id, mode):
        filepath = os.path.join(self.csv_folder, 'user{}_{}_EEG.csv'.format(id, mode))
        df = pd.read_csv(filepath,sep=',', low_memory=False)
        eeg = df.values
        if(mode == 'main'):
            Thought, RT, Rating, Trigger = eeg[:,-1], eeg[:,-2], eeg[:, -3], eeg[:, -4]
            eeg = np.delete(eeg, -1, axis = 1) # delete Thought
            eeg = np.delete(eeg, -1, axis = 1) # delete Rating Time
            eeg = np.delete(eeg, -1, axis = 1) # delete Rating
            eeg = np.delete(eeg, -1, axis = 1) # delete Trigger
            eeg = np.delete(eeg, 0 , axis = 1) # delete index
        else:
            Trigger = eeg[:, -1]
            eeg = np.delete(eeg, -1, axis = 1) # delete Trigger
            eeg = np.delete(eeg, 0 , axis = 1) # delete index
        eeg = np.transpose(eeg.astype('float')) # 33 x # of sample

        eeg = eeg - 0.5 * eeg[np.where(channel_list_M2 == 'M2')[0][0]] # re-reference
        eeg = np.delete(eeg, np.where(channel_list_M2 == 'M2')[0][0], axis = 0) # delete reference channel (32 x # of sample)
        

        if(mode == 'main'): # main experiment
            return eeg, Thought, RT, Rating, Trigger
        else: # prerest or post rest
            return eeg, 0, 0, 0, Trigger
    
    def bandpass(self, x):
        '''
            bandpass filter from 0.1 Hz to 45 Hz
        '''
        def _butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a
        def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y
        fs = 1000.0
        lowcut = 0.1
        highcut = 45.0
    
        x_f = []
        for channel in range(len(x)):
            x_f.append(_butter_bandpass_filter(x[channel], lowcut, highcut, fs, order= 2))
        return np.array(x_f)

    def plot_IC_component(self, ica, raw, _id, mode, saved_loc):
        k = 0
        for i in range(6):
            for j in range(5):
                #if(((i == 0) & (j == 0)) | ((i == 3) & (j == 0))):
                if((i == 0) & (j == 0)):
                    plt.figure(figsize = (19,10))
                    l = 1
                plt.subplot(6,5,l).set_title('ICA {}'.format(k))
                plt.plot(ica.get_sources(raw).get_data()[k])
                k += 1
                l += 1
        plt.tight_layout()
        plt.savefig(saved_loc)

    def ica(self, eeg, _id, mode):
        eeg_10s = eeg[:, -10000:] # last 10 seconds EEG
        info = mne.create_info(channel_list, 1000, ch_types=["eeg"] * 30 + ["eog"] * 2) 
        # sampling frequency of 1000 Hz and the channel types
        raw = mne.io.RawArray(eeg, info)
        raw_10s = mne.io.RawArray(eeg_10s, info)
        
        raw.set_montage("standard_1020") 
        # standard 10–20 montage, which maps the channel labels to their positions on the scalp.
    
        raw_tmp = raw.copy()
        #raw_tmp.filter(1, None) # apply a high-pass filter to this copy. filter < 1 Hz signals
    
        ica = mne.preprocessing.ICA(method="fastica", random_state=1)
        ica.fit(raw_tmp)
        
        self.plot_IC_component(ica, raw.copy(), _id, mode, saved_loc = os.path.join(SAVED_PATH, 'npy/ica/{}_fig/whole_proj_{}_ic_comp'.format(mode, _id)))
        self.plot_IC_component(ica, raw_10s.copy(), _id, mode, saved_loc = os.path.join(SAVED_PATH, 'npy/ica/{}_fig/10s_proj_{}_ic_comp'.format(mode, _id)))
        
        ica.plot_components(inst=raw_tmp, show = False, picks = np.arange(30)) # plot 22 components, you have to observe where the IC component is and remove it
        plt.savefig(os.path.join(SAVED_PATH, 'npy/ica/{}_fig/ic_comp_{}'.format(mode, _id)))

        raw_corrected = raw.copy()


        # remove IC component, you have to check where the IC component is, it is not always at index 0
        if(mode == 'main'):
            if(_id == '04'): removed_idx = 4
            elif(_id == '09'): removed_idx = 1
            elif(_id == '42'): removed_idx = 2
            elif(_id == '43'): removed_idx = 1
            elif(_id == '47'): removed_idx = 1
            elif(_id == '56'): removed_idx = 1
            else: removed_idx = 0
            raw_corrected = ica.apply(raw_corrected, exclude = [removed_idx])
            print(_id, removed_idx)
        elif(mode == 'pre'):
            removed_idx = 0
            raw_corrected = ica.apply(raw_corrected, exclude = [removed_idx])
            print(_id, removed_idx)
        elif(mode == 'post'):
            removed_idx = 0
            raw_corrected = ica.apply(raw_corrected, exclude = [removed_idx])
            print(_id, removed_idx)
        else:
            raw_corrected = ica.apply(raw_corrected, exclude = [0])
        
        return raw_corrected.get_data()

    def main_from_scratch(self, id_list, mode):
        '''
            mode: pre = pre-rest
                  post = post-rest
                  main = main experiment
            id_list: processed users
        '''
        for _id in id_list:
            print('Start Process Subject: {}'.format(_id))
            eeg, Thought, RT, Rating, Trigger = self.csv2numpy(id = _id, mode = mode) # re-reference
            eeg = self.bandpass(eeg) # bandpass filter

            if(mode == 'main'):
            	if(EEG_TYPE == 'Clean'):
            		eeg = self.ica(eeg, _id, mode = mode)
            	dic = {}
            	dic['EEG'] = eeg
            	dic['Thought'] = Thought
            	dic['RT'] = RT
            	dic['Rating'] = Rating
            	dic['Trigger'] = Trigger
	
            	if(EEG_TYPE == 'Dirty'):
            		np.save(os.path.join(SAVED_PATH, 'npy/no_ica/{}/{}_{}_DirtyEEG.npy'.format(mode, _id, mode)), dic)
            	elif(EEG_TYPE == 'Clean'):
            	    np.save(os.path.join(SAVED_PATH, 'npy/ica/{}/{}_{}_CleanEEG.npy'.format(mode, _id, mode)), dic)
            elif(mode == 'pre'):
            	close_start_idx = np.where(Trigger == 'target')[0][0]
            	close_end_idx = np.where(Trigger == 'target_response')[0][0]
            	open_start_idx = np.where(Trigger == 'normal')[0][0]
            	open_end_idx = np.where(Trigger == 'response')[0][0]
            	print('Duration of close eyes: {:.1f}s, Duration of open eyes: {:.1f}'
            		.format((close_end_idx - close_start_idx)/1000, (open_end_idx - open_start_idx)/1000))
            	close_eeg = eeg[:, close_start_idx:close_end_idx]
            	open_eeg = eeg[:, open_start_idx:open_end_idx]

            	if(EEG_TYPE == 'Clean'):
            		open_eeg = self.ica(eeg, _id, mode = mode)
            	dic = {}
            	dic['open_eeg'] = open_eeg
            	dic['close_eeg'] = close_eeg
            	dic['Trigger'] = Trigger

            	if(EEG_TYPE == 'Dirty'):
            		np.save(os.path.join(SAVED_PATH, 'npy/no_ica/{}/{}_{}_DirtyEEG.npy'.format(mode, _id, mode)), dic)
            	elif(EEG_TYPE == 'Clean'):
            		np.save(os.path.join(SAVED_PATH, 'npy/ica/{}/{}_{}_CleanEEG.npy'.format(mode, _id, mode)), dic)
            elif(mode == 'post'):
                close_start_idx = np.where(Trigger == 'target')[0][0]
                close_end_idx = np.where(Trigger == 'target_response')[0][0]
                open_start_idx = np.where(Trigger == 'normal')[0][0]
                open_end_idx = np.where(Trigger == 'response')[0][0]
                print('Duration of close eyes: {:.1f}s, Duration of open eyes: {:.1f}'
                    .format((close_end_idx - close_start_idx)/1000, (open_end_idx - open_start_idx)/1000))

def make_directory():
    if not os.path.exists(os.path.join(SAVED_PATH, 'npy/ica/main')):
        os.makedirs(os.path.join(SAVED_PATH, 'npy/ica/main'))
        os.makedirs(os.path.join(SAVED_PATH, 'npy/ica/main_fig'))# use to check where is the eye component
    if not os.path.exists(os.path.join(SAVED_PATH, 'npy/ica/post')):
        os.makedirs(os.path.join(SAVED_PATH, 'npy/ica/post'))
        os.makedirs(os.path.join(SAVED_PATH, 'npy/ica/post_fig'))# use to check where is the eye component
    if not os.path.exists(os.path.join(SAVED_PATH, 'npy/ica/pre')):
        os.makedirs(os.path.join(SAVED_PATH, 'npy/ica/pre'))
        os.makedirs(os.path.join(SAVED_PATH, 'npy/ica/pre_fig')) # use to check where is the eye component

    if not os.path.exists(os.path.join(SAVED_PATH, 'npy/no_ica/main')):
        os.makedirs(os.path.join(SAVED_PATH, 'npy/no_ica/main'))
    if not os.path.exists(os.path.join(SAVED_PATH, 'npy/no_ica/post')):
        os.makedirs(os.path.join(SAVED_PATH, 'npy/no_ica/post'))
    if not os.path.exists(os.path.join(SAVED_PATH, 'npy/no_ica/pre')):
        os.makedirs(os.path.join(SAVED_PATH, 'npy/no_ica/pre'))
if __name__ == '__main__':
    make_directory()
    s_time = time.time()
    id_list = ['48']
    main = csv_to_npy()
    main.main_from_scratch(id_list = id_list, mode = 'main')
    #main.main_from_scratch(id_list = id_list, mode = 'pre')
    #main.main_from_scratch(id_list = id_list, mode = 'post')
    #main.main_add_subject(add_id_list = add_id_list, mode = 'main')

    print('Used time: {:.2f}s'.format(time.time() - s_time))
