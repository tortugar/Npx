# Basic Neuropixels Analysis

## An example recording can be found under: 

The folder Data contains a text file (`mouse_config.txt`) defining where all data files for each mouse are located and one example Neuropixels recording of mouse DL159.

## Data organization

### Neuropixels data

The file `Data/mouse_config.txt` define for each recording where all relevant data are located, using the following format:

```
MOUSE: DL176
SL_PATH: /Volumes/T8/Data/Neuropixels/DL159/DL176_16_061322n3
NP_PATH: /Volumes/T8/Data/Neuropixels/DL176
TR_PATH: /Volumes/T8/Data/Neuropixels/DL176
KCUT: 0-660;8500-$
```

These lines specific for mouse DL176 that the sleep data are in the folder mentioned after `SL_PATH`. 
The files with all the firing rate information and histology alignment are in the folder `NP_PATH`. The spike trains discretized in 1 ms bins are saved under the folder `TR_PATH`.
Using `KCUT`, we can define time intervals that should be cut out from the recording. In this case, seconds 0-660 should be removed and the time interval from 8500s till the end of the recording (`$`). 

Under `EXCLUDE: ` you can exclude units for further analysis. 

Note: You can only remove intervals at the beginning or the end of the recording.

### EEG/EMG recording


