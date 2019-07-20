## To synchronize data
Processing raw data to csv file and do the synchronization
```
python3 merge_EEG_trigger.py
```
The output csv file is be like as follows

|TIMESTAMP|FP1|FP2|...|HEO|VEO|trigger|Rating|Rating_RT|Thought|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| time stamp 1|...|...|...|...|...|...|...|...|...|
| time stamp 2|...|...|...|...|...|...|...|...|...|
| time stamp 3|...|...|...|...|...|...|...|...|...|



The subject id whose data can't be used: 11, 13, 22, 27, 36, 72

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


