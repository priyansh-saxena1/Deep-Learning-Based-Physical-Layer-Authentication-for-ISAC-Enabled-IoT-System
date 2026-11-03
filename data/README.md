# Data

No dataset files are stored in this repository.

The experiments use the **DISC** dataset (60 GHz IEEE 802.11ay channel impulse
responses, 7 subjects, four activities: walking, running, sitting down / standing up,
waving hands), hosted on IEEE DataPort:

* https://ieee-dataport.org/documents/disc-dataset-integrated-sensing-and-communication-mmwave-systems
* DOI 10.21227/2gm7-9z72

DataPort requires an account and subscription to download. Please get the files from
there and follow their terms.

## What to put where

From the downloaded archive, copy the micro-Doppler spectrograms so the folder looks like:

```
data/
  micro_doppler_stft/
    HANDS_1_1.npy
    RUNNING_1_1.npy
    SITTING_1_1.npy
    WALKING_1_1.npy
    ...
```

File names are `<ACTIVITY>_<subject>_<index>.npy`. In our copy there were 433 files; one
(`SITTING_3_12.npy`) is entirely NaN and is skipped.

Then build the model input:

```bash
python preprocess_md.py
```

This writes `data/preprocessed_dataset/preprocessed_md.h5` with `(432, 64, 64, 1)`
samples, plus labels, subject ids and activity names. Expected console output:

```
built (432, 64, 64, 1)
NaN time-frames dropped: 199 ; files skipped: 1 ['SITTING_3_12.npy']
labels    : {0: 133, 1: 299}
```

The raw `.mat` CIR files (about 68 GB) are only needed for the older
`data_preprocessing.py` pipeline.
