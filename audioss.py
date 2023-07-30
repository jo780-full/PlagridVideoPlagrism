import librosa
import numpy as np
import torch
from torchvision.models import resnet50
from torchvision.transforms import functional as F
import os

def calculate_similarity(audio_path_1, audio_path_2):
    # Load audio files
    audio_data_1, sr_1 = librosa.load(audio_path_1)
    audio_data_2, sr_2 = librosa.load(audio_path_2)

    # Extract audio features using librosa
    features_1 = extract_features(audio_data_1, sr_1)
    features_2 = extract_features(audio_data_2, sr_2)

    # Calculate similarity score
    similarity_score = calculate_cosine_similarity(features_1, features_2)

    return similarity_score

def extract_features(audio_data, sr):
    # Extract audio features using librosa
    # For example, you can use Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
    features = torch.from_numpy(mfcc.T).float()

    return features

def calculate_cosine_similarity(features_1, features_2):
    # Resize or reshape feature tensors to match dimensions
    if features_1.shape[0] != features_2.shape[0]:
        min_length = min(features_1.shape[0], features_2.shape[0])
        features_1 = features_1[:min_length]
        features_2 = features_2[:min_length]

    # Calculate similarity score using cosine similarity
    similarity_matrix = torch.nn.functional.cosine_similarity(features_1, features_2, dim=1)
    average_similarity = similarity_matrix.mean().item()

    return average_similarity

with open('audio.txt', 'r') as file:
    lines = file.readlines()

if len(lines) >= 1:
    audio_1 = './' + lines[0].strip()

if len(lines) >= 2:
    audio_2 = './' + lines[1].strip()

# Check if audio_1 exists
if not os.path.isfile(audio_1):
    print(f"Audio file {audio_1} does not exist.")
    exit()

# Check if audio_2 exists
if not os.path.isfile(audio_2):
    print(f"Audio file {audio_2} does not exist.")
    exit()

# Calculate similarity score
similarity_score = calculate_similarity(audio_1, audio_2)
print(f"Similarity Score: {similarity_score}")
with open('similar.txt', 'a') as file:
    file.write(f"Similarity Score for text: {similarity_score}"+'\n')