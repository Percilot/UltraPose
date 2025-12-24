import os
import re
import cv2
import numpy as np
from scipy.ndimage import zoom
import scipy.io as sio
import librosa
from scipy.stats import wasserstein_distance
from itertools import combinations
import argparse

def compute_stft_features(signal, n_fft=1024, hop_length=512):
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(D)
    return magnitude

def compute_velocity_embedding(stft1, stft2):
    T = stft1.shape[1]
    velocity_embedding = []
    
    for t in range(T):
        frame1 = stft1[:, t]
        frame2 = stft2[:, t]

        frame1_mag = np.abs(frame1).flatten()
        frame2_mag = np.abs(frame2).flatten()

        distance = wasserstein_distance(frame1_mag, frame2_mag)
        velocity_embedding.append(distance)
    
    velocity_embedding = np.array(velocity_embedding)
    
    return velocity_embedding

def compute_delay_centroid(D):
    T, L = D.shape
    centroids = np.zeros(T)
    
    for t in range(T):
        tau = np.arange(L)
        weighted_sum = np.sum(tau * D[t])
        total_sum = np.sum(D[t])
        centroids[t] = weighted_sum / total_sum if total_sum != 0 else 0
        
    return centroids

def compute_distance_embedding(centroids1, centroids2):
    distance_embedding = centroids2 - centroids1
    return distance_embedding

def find_closest_rectangle(n):
    best_h, best_w = 1, n
    min_diff = n

    for h in range(1, int(np.sqrt(n)) + 1):
        if n % h == 0:
            w = n // h
            diff = abs(h - w)
            if diff < min_diff:
                best_h, best_w = h, w
                min_diff = diff

    return best_h, best_w

def rearrange_and_resize(matrix, target_shape=(128, 128)):
    total_elements = matrix.shape[0] * matrix.shape[1]
    h, w = find_closest_rectangle(total_elements)

    flat = matrix.flatten()
    reshaped = flat.reshape((h, w))

    resized = cv2.resize(reshaped, target_shape, interpolation=cv2.INTER_LINEAR)
    return resized


def process_mat_files(input_folder, output_folder, n, label, label_mag, label_ang):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]
    if not mat_files:
        print("No .mat files found in the input folder.")
        return
    
    print(f"Found {len(mat_files)} .mat files to process.")
    
    signals = {}
    for mat_file in mat_files:
        mat_data = sio.loadmat(os.path.join(input_folder, mat_file))
        signal_mag = mat_data.get(label_mag)
        signal_ang = mat_data.get(label_ang)
        
        if signal_mag is not None and signal_ang is not None:
            signals[mat_file] = (signal_mag, signal_ang)
    
    if not signals:
        print("No valid signals found in the .mat files.")
        return
    
    for (file1, file2) in combinations(signals.keys(), 2):
        signal1_mag, signal1_ang = signals[file1]
        signal2_mag, signal2_ang = signals[file2]
        
        centroids1 = compute_delay_centroid(signal1_mag)
        centroids2 = compute_delay_centroid(signal2_mag)
        distance_embedding = compute_distance_embedding(centroids1, centroids2)
        
        stft1 = compute_stft_features(signal1_mag)
        stft2 = compute_stft_features(signal2_mag)
        velocity_embedding = compute_velocity_embedding(stft1, stft2)

        source_mag_resized = rearrange_and_resize(signal1_mag)
        source_ang_resized = rearrange_and_resize(signal1_ang)
        target_mag_resized = rearrange_and_resize(signal2_mag)
        target_ang_resized = rearrange_and_resize(signal2_ang)
        combined_source = np.stack([source_mag_resized, source_ang_resized], axis=0)
        combined_target = np.stack([target_mag_resized, target_ang_resized], axis=0)


        file1_number = re.search(r'\d+', file1).group()
        file2_number = re.search(r'\d+', file2).group()
        
        source_folder = os.path.join(output_folder, "source")
        target_folder = os.path.join(output_folder, "target")
        condition_d_folder = os.path.join(output_folder, "condition_d")
        condition_v_folder = os.path.join(output_folder, "condition_v")
        
        os.makedirs(source_folder, exist_ok=True)
        os.makedirs(target_folder, exist_ok=True)
        os.makedirs(condition_d_folder, exist_ok=True)
        os.makedirs(condition_v_folder, exist_ok=True)
        
        source_path = os.path.join(source_folder, f"{label}_{file1_number}_{file2_number}.npy")
        target_path = os.path.join(target_folder, f"{label}_{file1_number}_{file2_number}.npy")
        cond_d_path = os.path.join(condition_d_folder, f"{label}_{file1_number}_{file2_number}.npy")
        cond_v_path = os.path.join(condition_v_folder, f"{label}_{file1_number}_{file2_number}.npy")
        
        np.save(source_path, combined_source)
        np.save(target_path, combined_target)
        np.save(cond_d_path, distance_embedding)
        np.save(cond_v_path, velocity_embedding)
        
        print(f"Saved: {source_path}, {target_path}, {cond_d_path}, {cond_v_path}")

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process .mat files to generate and merge distance and velocity embeddings")
    parser.add_argument('--input', type=str, required=True, help="Input folder containing .mat files")
    parser.add_argument('--output', type=str, required=True, help="Output folder to save .npy files")
    parser.add_argument('--n', type=int, required=True, help="Size of the square matrix (n x n)")
    parser.add_argument('--label', type=str, required=True, help="Label for the dataset")
    parser.add_argument('--label_mag', type=str, required=True, help="Matrix name to extract magnitude data from .mat files")
    parser.add_argument('--label_ang', type=str, required=True, help="Matrix name to extract angle data from .mat files")
    return parser.parse_args()

# Main program
if __name__ == "__main__":
    args = parse_args()
    input_folder = args.input
    output_folder = args.output
    n = args.n
    label = args.label
    label_mag = args.label_mag
    label_ang = args.label_ang
    process_mat_files(input_folder, output_folder, n, label, label_mag, label_ang)
