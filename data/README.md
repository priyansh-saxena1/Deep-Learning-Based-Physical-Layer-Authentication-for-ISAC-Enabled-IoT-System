# Data

Everything here comes from the **DISC** dataset (60 GHz IEEE 802.11ay channel impulse
responses, 7 subjects, four activities: walking, running, sitting down / standing up,
waving hands). Authors, DOI and the CC BY 4.0 terms are in [`../DATA_LICENSE.md`](../DATA_LICENSE.md).

```
data/
  micro_doppler_stft/                 433 spectrograms, unmodified from DISC
  preprocessed_dataset/
    preprocessed_md.h5                (432, 64, 64, 1) model input built from them
```

File names are `<ACTIVITY>_<subject>_<index>.npy`. One file, `SITTING_3_12.npy`, is
entirely NaN and is skipped, leaving 432 samples (133 with label 0, 299 with label 1).

To rebuild the `.h5` from the spectrograms:

```bash
python preprocess_md.py
```

Expected console output:

```
built (432, 64, 64, 1)
NaN time-frames dropped: 199 ; files skipped: 1 ['SITTING_3_12.npy']
labels    : {0: 133, 1: 299}
```

The raw `.mat` CIR files (about 68 GB) are not included and are only needed for the older
`data_preprocessing.py` pipeline. Get them from IEEE DataPort:
https://ieee-dataport.org/documents/disc-dataset-integrated-sensing-and-communication-mmwave-systems
