## To synchronize data
Processing raw data (in the **FORMAL_DATA_PATH**) to csv file and do the synchronization, save the file to **USER_CSV_PATH**
```
python3 synchronize.py
```
1. The output **main** csv file is be like as follows

|TIMESTAMP|FP1|FP2|...|HEO|VEO|trigger|Rating|Rating_RT|Thought|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| time stamp 1|...|...|...|...|...|...|...|...|...|
| time stamp 2|...|...|...|...|...|...|...|...|...|
| time stamp 3|...|...|...|...|...|...|...|...|...|

 * Type of **thought**
   * normal
   * probe
   * response
   * target
 * When the type of **thought** == target, the **response time, rating and trigger** will have value


2. The output **rest** csv file is be like as follows

|TIMESTAMP|FP1|FP2|...|HEO|VEO|trigger|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| time stamp 1|...|...|...|...|...|...|
| time stamp 2|...|...|...|...|...|...|
| time stamp 3|...|...|...|...|...|...|


The subject id whose data can't be used: 11, 13, 22, 27, 36, 72

## Preprocessing of EEG 
### Re-reference -> Bandpass filter -> ICA eye artifact rejection (optional) -> Save to npz file
```
python3 preprocess.py -p -ica
  -p Preprocess the raw data (Re-reference -> Bandpass filter)
  -ica Remove eye artifact use ICA (ICA eye artifact rejection)
```
* ***ONLY SUPPORT PREPROCESS NOW***
* If using ICA
  * Need to check the IC component index of eye blinking
    * variable: **ICA_EXCLUDED_COMPONENTS** in the preprocess.py
* **mode**
  * **main**: Save the npz file with keys (eeg, thought, response_time, rating, trigger)
  * **pre** or **post**: Save the npz file with keys (open_eeg, close_eeg)

## Merge each subject's epoched data into one file
```
python3 epoch.py
  -bp epoch eeg data 10s before probe (TRUE), or epoch eeg data 10s before C appears (FALSE) in the main experiment
  -ica Remove eye artifact use ICA
```
