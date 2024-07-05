# Basic Neuropixels Analysis

## Plotting firing rates and PCA

(1) In a text file (`mouse_config.txt`) define for each recording where all relevant data is located,
using the following format:

```
MOUSE: DL176
SL_PATH: /Volumes/T8/Data/Neuropixels/DL176/DL176_16_061322n3
NP_PATH: /Volumes/T8/Data/Neuropixels/DL176
TR_PATH: /Volumes/T8/Data/Neuropixels/DL176
KCUT: 0-660;8500-$
```

These lines specific for mouse DL176 that the sleep data in the folder after `SL_PATH:`. 
The files with all the firing rate information and histology alignment are in the folder `NP_PATH`. The ms spike trains are saved under the folder `TR_PATH`.
Using `KCUT:`, we can define time intervals that should be cut out from the recording. In this case, seconds 0-660 should be removed and the time interval from 8500s till the end of the recording (`$`). 

Under `EXCLudE: ` you can exclude units for further analysis. 

Note: You can only remove intervals at the beginning of the end of the recording.
