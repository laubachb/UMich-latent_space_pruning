"""
Process raw ChIMES descriptor output (A.txt + natoms.txt) into a pickle file
used as input to compute_and_prune.py.

Run from this directory:
    python process_raw_descriptors.py

Expects A.txt and natoms.txt to be present in the working directory.
A.txt format: each frame block = 3*natoms descriptor rows + 3 metadata lines.
natoms.txt format: one integer per line, giving atom count for each frame.

Output: frames_descriptors.pkl  — dict mapping frame_index -> np.ndarray (natoms, n_features)
"""

import numpy as np
import pickle


def process_to_descriptor_dict(a_path='A.txt', natoms_path='natoms.txt'):
    with open(a_path, 'r') as f:
        a_lines = f.readlines()

    with open(natoms_path, 'r') as f:
        natoms_lines = f.readlines()

    natoms_list = [int(line.strip()) for line in natoms_lines]

    index = 0
    frame = 0
    data_dict = {}
    frames_with_nan = []

    while index < len(a_lines):
        try:
            natoms = natoms_list[index]
        except IndexError:
            print(f"Index {index} out of range in natoms_list.")
            break

        # Each frame block: 3*natoms descriptor rows + 3 metadata lines
        chunk_size = 3 * natoms + 3

        if index + chunk_size > len(a_lines):
            print(f"Incomplete data for frame {frame}. Stopping.")
            break

        descriptor_lines = a_lines[index: index + 3 * natoms]

        try:
            matrix = np.array([
                list(map(float, line.strip().split()))
                for line in descriptor_lines
            ])
        except ValueError as e:
            print(f"Error parsing frame {frame} at index {index}: {e}")
            break

        if np.isnan(matrix).any():
            nan_count = int(np.isnan(matrix).sum())
            print(f"Warning: frame {frame} contains {nan_count} NaN values.")
            frames_with_nan.append(frame)

        data_dict[frame] = matrix
        index += chunk_size
        frame += 1

    print(f"\nProcessed {frame} frames total.")
    if frames_with_nan:
        print(f"Warning: {len(frames_with_nan)} frames contain NaN values: {frames_with_nan}")
    else:
        print("No NaN values detected.")

    return data_dict


def save_dict_to_pickle(data_dict, output_path='frames_descriptors.pkl'):
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Saved descriptor dictionary to '{output_path}'.")


if __name__ == "__main__":
    descriptor_dict = process_to_descriptor_dict()
    save_dict_to_pickle(descriptor_dict)
