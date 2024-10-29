# Basic Neuropixels Analysis

## An example recording can be found under: 

The folder Data contains a text file (`mouse_config.txt`) defining where all data files for each mouse are located and one example Neuropixels recording of mouse DL159.

## Data organization

### Neuropixels data

The file `Data/mouse_config.txt` define for each recording where all relevant data are located, using the following format:

```
MOUSE: DL159
SL_PATH: /Volumes/T8/Data/Neuropixels/DL159/
NP_PATH: /Volumes/T8/Data/Neuropixels/DL159
TR_PATH: /Volumes/T8/Data/Neuropixels/DL159
KCUT: 0-660;8500-$
```

`MOUSE` defines the mouse name. 

`NP_PATH` defines the folder containing all Neuropixels-related data, including ...
  *  `traind.csv` is a pandas DataFrame (pd.DataFrame). Each columns corresponds to a unit (the columns name is the unit ID), each row corresponds to a 2.5 s time bin. The binning is aligned with the sleep annotation, EEG/EMG spectrograms (in `SL_PATH`).
  *  `1ktrain.npz`: Spike trains (encoded as 1s and 0s) for each unit with 1ms resolution.
  *  `channel_locations.json` describes for each unit (referenced by its ID) the brain region and location within the Allen 3D Brain Atlas. 

The folder `TR_PATH` contains the file `1ktrain.npz` with the spike trains (encoded as 1s and 0s) for each unit in 1ms resolution.
  
Using `KCUT`, we can define time intervals at the beginning or end of the recording that should be cut out from the recording (often related to drift, or the mouse needing some time to fall asleep). In this case, seconds 0-660 should be removed and the time interval from 8500s till the end of the recording (`$`). 

Using `EXCLUDE` you can exclude units from further analysis (e.g. units with strong drift in their firing rates). 


### Sleep data

The folder `SL_PATH` contains all sleep-related files including ...
* The raw EEG and EMG sampled using 1 kHz resolution.
* The sleep annotation `remidx_*.txt` in 2.5 s resolution. `*` is the name of the sleep recording (in our example, that's `DL159_7721n1`. 
* The EEG and EMG spectrogram using the same 2.5 s binning, saved under `sp_*.mat` and `spm_*.mat`.
* `info.txt` contains some basic information about the sleep recording (amplifier, sampling rate, time of recording, duration of recording).

---

# System requirements 

* All code is written in Python 3, and has been tested on Python versions 3.7, and 3.9.

* The required packages for Python scripts/modules are listed at the beginning of each file. Packages can be installed using the conda package manager in the Anaconda distribution of Python. Go to https://www.anaconda.com/ to install Anaconda on Windows, MacOS, or Linux. Installation will take approximately 15 minutes.

* To run our code, our modules need to be added the python's system path, which can be done using

```
import sys
sys.path.add([path to module folder])
```

