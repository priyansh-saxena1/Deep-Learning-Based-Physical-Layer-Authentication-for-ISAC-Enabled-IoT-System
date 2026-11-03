import os
import cupy as cp
import numpy as np
import h5py
import random
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.pyplot as plt
from config import DATASET_DIR, OUTPUT_DIR, LABEL_MAPPING, PADDING_MULTIPLE, BATCH_SIZE, TRAIN_TEST_SPLIT

def preprocess_data(DATASET_DIR, OUTPUT_DIR, LABEL_MAPPING, BATCH_SIZE):
    # Step 1: Determine the target dimension for all data points
    # dimensions = []
    # file_list = [f for f in os.listdir(DATASET_DIR) if f.endswith(".mat")]

    # for file_name in file_list:
    #     file_path = os.path.join(DATASET_DIR, file_name)
    #     mat_data = sio.loadmat(file_path)
    #     matrix = mat_data["FRAMES"][0][0]["CIR"]
    #     dimensions.append(matrix.shape[1])  # Variable dimension across samples

    # # Compute target dimension (95th percentile) and ensure divisibility by 8
    # D_max = max(dimensions)
    # D_target = int(np.percentile(dimensions, 95))
    # D_target = ((D_target + 63) // 64) * 64  # Round up to nearest multiple of 64

    # Step 1: Determine the target dimension for all data points
    dimensions = []
    file_list = [f for f in os.listdir(DATASET_DIR) if f.endswith(".mat")]

    for file_name in file_list:
        file_path = os.path.join(DATASET_DIR, file_name)
        mat_data = sio.loadmat(file_path)
        matrix = mat_data["FRAMES"][0][0]["CIR"]
        dimensions.append(matrix.shape[1])  # Variable dimension across samples

    # Compute min and max dimensions
    D_min = min(dimensions)
    D_max = max(dimensions)

    # Create histogram
    plt.figure(figsize=(8, 5))
    bins = np.arange(D_min, D_max + 32, 32)  # Bin size of 32
    hist_values, bin_edges, _ = plt.hist(dimensions, bins=bins, edgecolor='black', alpha=0.7)

    # Identify the bin with the highest count
    most_frequent_bin_index = np.argmax(hist_values)
    most_frequent_range = (bin_edges[most_frequent_bin_index], bin_edges[most_frequent_bin_index + 1])
    D_target = int(most_frequent_range[1])  # Round up to nearest multiple of 32
    D_target = (D_target // 32) * 32  # Ensure divisibility by 32

    # Plot enhancements
    plt.xlabel("Matrix Dimension")
    plt.ylabel("Number of Matrices")
    plt.title("Distribution of Matrix Dimensions")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()

    # Print the updated target dimension
    print(f"Min Dimension: {D_min}, Max Dimension: {D_max}")
    print(f"Target Dimension: {D_target}")

    # Downsample dimensions
    D_downsampled = D_target // 4
    C_downsampled = 512 // 4

    print(f"Max dimension: {D_max}, Target dimension: {D_target}, Downsampled dimensions: {D_downsampled}, {C_downsampled}")
    test_label_0_count = 0
    test_label_1_count = 0

    # Step 2: Create an HDF5 file for batch storage
    with h5py.File(os.path.join(OUTPUT_DIR, "preprocessed_data.h5"), 'w') as hf:
        # Create groups for train and test
        hf.create_group('train')
        hf.create_group('test')

        # Create datasets with expandable dimensions (batch processing)
        for split in ["train", "test"]:
            hf[split].create_dataset('data', (0, D_downsampled, C_downsampled, 2), maxshape=(None, D_downsampled, C_downsampled, 2), dtype='f4', compression="gzip")
            hf[split].create_dataset('labels', (0,), maxshape=(None,), dtype='i4', compression="gzip")

        # Store preprocessing metadata
        hf.attrs['preprocessing_info'] = 'Normalization, padding, downsampling, and conversion to image format applied'
        hf.attrs['train_test_split_ratio'] = '80:20'


        # Step 3: Process files in batches and write directly to disk
        data_batch = {"train": [], "test": []}
        labels_batch = {"train": [], "test": []}

        for idx, file_name in enumerate(file_list, start=1):
            print(f"Processing file {idx}/{len(file_list)}: {file_name}")
            file_path = os.path.join(DATASET_DIR, file_name)
            mat_data = sio.loadmat(file_path)
            matrix = mat_data["FRAMES"][0][0]["CIR"]
            matrix_gpu = cp.array(matrix)  # Load data onto GPU

            # Normalize along the last two dimensions (variable, 512)
            matrix_min = cp.min(matrix_gpu, axis=(1, 2), keepdims=True)
            matrix_max = cp.max(matrix_gpu, axis=(1, 2), keepdims=True)

            # Perform Min-Max normalization dimension-wise
            matrix_gpu_norm = (matrix_gpu - matrix_min) / (matrix_max - matrix_min + 1e-8)

            # Padding or truncation
            current_dim = matrix_gpu_norm.shape[1]
            if D_target > current_dim:
                pad_width = ((0, 0), (0, D_target - current_dim), (0, 0))
                matrix_gpu_padded = cp.pad(matrix_gpu_norm, pad_width, mode='constant', constant_values=0)
            else:
                matrix_gpu_padded = matrix_gpu_norm[:, :D_target, :]

            # Convert complex data into a 2-channel image format
            image_format = cp.stack([cp.real(matrix_gpu_padded), cp.imag(matrix_gpu_padded)], axis=-1)
            image_format = image_format[0, :, :, :]  # Remove batch dimension

            # Downsample the data by a factor of 8
            if D_target % 4 == 0 and 512 % 4 == 0:
                downsampled = image_format.reshape(D_downsampled, 4, C_downsampled, 4, 2).mean(axis=(1, 3))
            else:
                print(f"Skipping {file_name} due to incompatible dimensions for downsampling.")
                continue

            if downsampled.shape != (D_downsampled, C_downsampled, 2):
                print(f"Skipping {file_name} due to incorrect shape after downsampling: {downsampled.shape}")
                continue

            # Assign label based on filename
            label = next((LABEL_MAPPING[key] for key in LABEL_MAPPING if file_name.startswith(key)), -1)
            if label == -1:
                print(f"Skipping {file_name} due to unknown label.")
                continue

            # Controlled assignment based on fixed test size
            if label == 0 and test_label_0_count < 28:
                split = "test"
                test_label_0_count += 1
            elif label == 1 and test_label_1_count < 59:
                split = "test"
                test_label_1_count += 1
            else:
                split = "train"
            data_batch[split].append(cp.asnumpy(downsampled))
            labels_batch[split].append(label)

            # Write batch to file once it reaches batch size
            if len(data_batch[split]) >= BATCH_SIZE:
                print(f"Writing {BATCH_SIZE} samples to {split} dataset.")
                try:
                    # Convert to NumPy arrays
                    data_batch_np = np.array(data_batch[split], dtype=np.float32)
                    labels_batch_np = np.array(labels_batch[split], dtype=np.int32)
                except ValueError as e:
                    print(f"Batch error: {e}. Skipping batch.")
                    data_batch[split].clear()
                    labels_batch[split].clear()
                    continue

                # Resize dataset and append new data
                hf[split]['data'].resize((hf[split]['data'].shape[0] + BATCH_SIZE, D_downsampled, C_downsampled, 2))
                hf[split]['labels'].resize((hf[split]['labels'].shape[0] + BATCH_SIZE,))

                # original=====================================================================================================================================================
                hf[split]['data'][-BATCH_SIZE:] = data_batch_np
                hf[split]['labels'][-BATCH_SIZE:] = labels_batch_np

                # hf[split]['data'][-len(data_batch[split]):] = data_batch_np
                # hf[split]['labels'][-len(data_batch[split]):] = labels_batch_np


                # Clear batch
                data_batch[split].clear()
                labels_batch[split].clear()

        # Step 4: Handle remaining data
        for split in ["train", "test"]:
            if data_batch[split]:
                print(f"Writing remaining {len(data_batch[split])} samples to {split} dataset.")
                try:
                    data_batch_np = np.array(data_batch[split], dtype=np.float32)
                    labels_batch_np = np.array(labels_batch[split], dtype=np.int32)
                except ValueError as e:
                    print(f"Final batch error: {e}. Skipping last batch.")
                    continue

                hf[split]['data'].resize((hf[split]['data'].shape[0] + len(data_batch[split]), D_downsampled, C_downsampled, 2))
                hf[split]['labels'].resize((hf[split]['labels'].shape[0] + len(data_batch[split]),))

                hf[split]['data'][-len(data_batch[split]):] = data_batch_np
                hf[split]['labels'][-len(data_batch[split]):] = labels_batch_np

    print(f"Preprocessed data stored in {OUTPUT_DIR}/preprocessed_data.h5")

    return D_downsampled, C_downsampled