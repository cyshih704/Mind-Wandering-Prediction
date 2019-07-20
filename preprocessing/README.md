## To synchronize data
```
python3 merge_EEG_trigger.py
```

## Preprocessing EEG 
### Re-reference -> bandpass filter -> ICA -> Save to npy
* If using ICA
  * Change variable **EEG_TYPE** to 'Clean'
  * You have to check the index of the eye component manually and then remove it. 
    * The default removed index is 0, but the eye component index is depeents on subject 
  * ***How to check the index of IC component:*** open the image ***npy/ica/main_fig/ic_comp_id.png***

* If not using ICA
 * Change variable **EEG_TYPE** to 'Dirty'

* Variable **id_list**: the subjects that you want to process 

* Only support **mode == 'main'**, have to modify the code when **mode == 'pre' or mode == 'post'**
```
python3 csv_to_npy.py
```

## Merge each subject data into one file
* Variable: probe
  * True: segment EEG data 10 seconds before the PROBE (saved to 10s_Dirty_seg_data)
  * False: segment EEG data 10 secondes before the TARGET (saved to 10s_Dirty_seg_data_wC)
* Variable: EEG_TYPE
  * 'Dirty': not use ICA
  * 'Clean': use ICA
* Variable: id_list
  * Subject id that will be merged
* Variable ls
  * Length of each segment (default 10 seconds)
```
python3 seg_data.py
```


