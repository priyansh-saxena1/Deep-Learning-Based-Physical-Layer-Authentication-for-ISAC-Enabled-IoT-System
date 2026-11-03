# Data licence and attribution

The files under `data/micro_doppler_stft/` are taken unmodified from the **DISC**
dataset, and `data/preprocessed_dataset/preprocessed_md.h5` is derived from them by
`preprocess_md.py`.

> Pegoraro, J., Saucedo, J., Lacruz, J. O., Rossi, M., Widmer, J.
> *DISC: a Dataset for Integrated Sensing and Communication in mmWave Systems.*
> IEEE DataPort, 2023. DOI: [10.21227/2gm7-9z72](https://dx.doi.org/10.21227/2gm7-9z72)
> https://ieee-dataport.org/documents/disc-dataset-integrated-sensing-and-communication-mmwave-systems

DISC is published on IEEE DataPort under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** licence
(https://creativecommons.org/licenses/by/4.0/). Under that licence the data may be
shared and adapted, including commercially, as long as the authors are credited, the
licence is linked, and changes are indicated.

## Changes made

* `data/micro_doppler_stft/*.npy`: the micro-Doppler spectrograms supplied with DISC,
  copied byte for byte.
* `data/preprocessed_dataset/preprocessed_md.h5`: derived from the files above. Frames
  that are entirely NaN are dropped, one fully-NaN recording (`SITTING_3_12`) is
  skipped, the time axis is resampled to 64 frames, the Doppler axis is padded from 59
  to 64 bins, and each recording gets a binary label. See `preprocess_md.py`.

The DISC authors have not reviewed or endorsed this repository.

## Scope

This licence covers the data only. The source code is under the MIT licence in
[`LICENSE`](LICENSE). IEEE DataPort hosts the original dataset and, per its page, asks
for a DataPort subscription to download it; the copy here is redistributed under the
CC BY 4.0 grant above. If you need the raw channel impulse responses (`.mat`, about
68 GB), get them from DataPort.
