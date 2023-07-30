import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import os
def calculate_similarity(text_path_1, text_path_2):
    # Load text files
    with open(text_path_1, 'r', encoding='utf-8') as file:
        text_1 = file.read()
    with open(text_path_2, 'r', encoding='utf-8') as file:
        text_2 = file.read()

    # Extract text features using TF-IDF vectorizer
    features_1, features_2 = extract_features(text_1, text_2)

    # Calculate similarity score
    similarity_score = calculate_cosine_similarity(features_1, features_2)

    return similarity_score

def extract_features(text_1, text_2):
    # Extract text features using TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform([text_1, text_2])

    # Convert features to PyTorch tensors
    features_1 = torch.from_numpy(features[0].toarray()).float()
    features_2 = torch.from_numpy(features[1].toarray()).float()

    return features_1, features_2

def calculate_cosine_similarity(features_1, features_2):
    # Calculate similarity score using cosine similarity
    similarity_matrix = torch.nn.functional.cosine_similarity(features_1, features_2, dim=1)
    average_similarity = similarity_matrix.mean().item()

    return average_similarity

# Specify the paths to the text files

with open('texts.txt', 'r') as file:
    lines = file.readlines()

if len(lines) >= 1:
    text_1 = './' + lines[0].strip()

if len(lines) >= 2:
    text_2 = './' + lines[1].strip()

# Check if text_1 exists
if not os.path.isfile(text_1):
    print(f"Text file {text_1} does not exist.")
    exit()

# Check if text_2 exists
if not os.path.isfile(text_2):
    print(f"Text file {text_2} does not exist.")
    exit()

# Calculate similarity score
similarity_score = calculate_similarity(text_1, text_2)
print(f"Similarity Score: {similarity_score}")
with open('similar.txt', 'a') as file:
    file.write(f"Similarity Score for text: {similarity_score}"+'\n')