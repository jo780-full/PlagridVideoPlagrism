import cv2
import numpy as np
import torch
from torchvision.models import resnet50
from torchvision.transforms import functional as F
import os

def calculate_similarity(video_path_1, video_path_2):
    # Load video clips
    video_clip_1 = cv2.VideoCapture(video_path_1)
    video_clip_2 = cv2.VideoCapture(video_path_2)

    # Extract frames and calculate feature vectors using ResNet50
    features_1 = extract_features(video_clip_1, frame_skip=20)  # Extract every 5th frame for accuracy but takes to long to run
    features_2 = extract_features(video_clip_2, frame_skip=20)

    # Calculate similarity score
    similarity_score = calculate_cosine_similarity(features_1, features_2)

    return similarity_score

def extract_features(video_clip, frame_skip=1):
    # Define the pre-trained CNN model for feature extraction
    cnn_model = resnet50(pretrained=True)
    cnn_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)

    # Extract frames and calculate feature vectors
    features = []
    transform = F.to_tensor

    frame_count = 0
    with torch.no_grad():
        while video_clip.isOpened():
            ret, frame = video_clip.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))

            # Convert frame to tensor and normalize
            frame_tensor = transform(frame).unsqueeze(0).to(device)

            # Extract features using the CNN model
            feature = cnn_model(frame_tensor)
            features.append(feature)

    video_clip.release()

    if len(features) == 0:
        raise ValueError("No frames found in the video.")

    features = torch.cat(features, dim=0)

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



with open('vid.txt', 'r') as file:
    lines = file.readlines()

if len(lines) >= 1:
    video_1 = './' + lines[0].strip()

if len(lines) >= 2:
    video_2 = './' + lines[1].strip()
# Check if video_1 exists
if not os.path.isfile(video_1):
    print(f"Video file {video_1} does not exist.")
    exit()
if not os.path.isfile(video_2):
    print(f"Video file {video_2} does not exist.")
    exit()
# Calculate similarity score
similarity_score = calculate_similarity(video_1, video_2)
print(f"Similarity Score: {similarity_score}")
with open('similar.txt', 'a') as file:
    file.write(f"Similarity Score for Video: {similarity_score}"+'\n')
