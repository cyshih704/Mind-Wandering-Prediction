## To synchronize data
Processing raw data to csv file and do the synchronization
```
python3 merge_EEG_trigger.py
```
The output main csv file is be like as follows

|TIMESTAMP|FP1|FP2|...|HEO|VEO|trigger|Rating|Rating_RT|Thought|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| time stamp 1|...|...|...|...|...|...|...|...|...|
| time stamp 2|...|...|...|...|...|...|...|...|...|
| time stamp 3|...|...|...|...|...|...|...|...|...|


The output rest csv file is be like as follows

|TIMESTAMP|FP1|FP2|...|HEO|VEO|trigger|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| time stamp 1|...|...|...|...|...|...|
| time stamp 2|...|...|...|...|...|...|
| time stamp 3|...|...|...|...|...|...|



The subject id whose data can't be used: 11, 13, 22, 27, 36, 72

## Preprocessing of EEG 
### Re-reference -> Bandpass filter -> ICA eye artifact rejection (optional) -> Save to npz file
```
python3 csv_to_npy.py
```
* main
  * Save the npz file with keys (eeg, thought, response_time, rating, trigger)
* rest
  * Save the npz file with keys (open_eeg, close_eeg)

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

