import flask
from flask import *
import os
import secrets
import subprocess
import webbrowser
import moviepy.editor as mp
import assemblyai as aai
import shutil
import ffmpeg
import time
from pydub import AudioSegment
import glob
import io
import cv2
import numpy as np
import torch
from torchvision.models import resnet50
from torchvision.transforms import functional as F
import time
import imdb
import requests
import webbrowser
import re
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
file_paths=""
video_file_path=""
nltk.download('stopwords')
nltk.download('punkt')
# Your API


aai.settings.api_key = "b7102fe722b24bb3beba4a56090fc680"
transcriber = aai.Transcriber()
english_subtitle_urls = []
english_downloads = []
app = Flask(__name__)
app.config["DEBUG"] = True
app.config['SECRET_KEY'] = secrets.token_hex(16)
FILE_PATH=""
user_agent_array = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
 
]
def calculate_cosine_similarity(features_1, features_2):
    # Calculate similarity score using cosine similarity
    similarity_matrix = torch.nn.functional.cosine_similarity(features_1, features_2, dim=1)
    average_similarity = similarity_matrix.mean().item()

    return average_similarity

def extract_features(video_clip):
    # Define the pre-trained CNN model for feature extraction
    cnn_model = resnet50(pretrained=True)
    cnn_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)

    # Extract frames and calculate feature vectors
    features = []
    transform = F.to_tensor

    with torch.no_grad():
        while video_clip.isOpened():
            ret, frame = video_clip.read()

            if not ret:
                break

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
def calculates_similarity(video_path_1, video_path_2):
    # Load video clips
    video_clip_1 = cv2.VideoCapture(video_path_1)
    video_clip_2 = cv2.VideoCapture(video_path_2)

    # Extract frames and calculate feature vectors using ResNet50
    features_1 = extract_features(video_clip_1)
    features_2 = extract_features(video_clip_2)

    # Calculate similarity score
    similarity_score = calculate_cosine_similarity(features_1, features_2)

    return similarity_score
def remove_punctuation_and_numbers(text):
    # Remove punctuation and numbers using regular expressions
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
     
    # Tokenize text into words
    word_tokens = word_tokenize(cleaned_text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in word_tokens if word not in stop_words]
    
    # Join words back into sentences
    preprocessed_text = ' '.join(filtered_words)
    
    return preprocessed_text


def calculate_similarity_score(text1, text2):
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate cosine similarity between the vectors
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/extractor", methods=["POST", "GET"])
def extractor():
    global file_paths
    global video_file_path
    if request.method == "POST":
        file = request.files.get('file')
        if file.filename.endswith(".mkv"):
            if not os.path.exists("pre_mp4"):
                os.mkdir("pre_mp4")
            file_path = os.path.join("pre_mp4", file.filename)
            file.save(file_path)
        else:
            if not os.path.exists("uploads"):
                os.mkdir("uploads")
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            video_file_path=file_path
            with open('vid.txt','a') as file:
                file.write(video_file_path+'\n')
     
    if file_path.endswith((".mp4", ".mov")):
        name, _ = os.path.splitext(file_path)
        audio = mp.AudioFileClip(file_path)
        name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join('audios', name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        audio_path = os.path.join(save_path, f'{name}.wav')
        with open('audio.txt','a') as file:
            file.write(audio_path+'\n')
        audio.write_audiofile(audio_path)
        transcript = transcriber.transcribe(audio_path)
        
        save_path = os.path.join('subtitles')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        subtitle_path = os.path.join(save_path, f'{name}.txt')
        with open('texts.txt','a') as file:
            file.write(subtitle_path+'\n')
        file_paths = subtitle_path
        if transcript.text is not None:
            with open(subtitle_path, 'w', encoding='utf-8') as file:
                file.write(transcript.text)

    # Convert MKV to MP4
    if file_path.endswith(".mkv"):
        name, _ = os.path.splitext(file_path)
        out_name = name + ".mp4"
        print("Converting from MKV to MP4...")
    
        # Use subprocess to run the ffmpeg command
        subprocess.run(['ffmpeg', '-i', file_path, out_name])
        search_pattern = os.path.join("pre_mp4", name + "*.mp4")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            # Found MP4 file(s) with the same name
            for mp4_file in matching_files:
                audio = mp.AudioFileClip(mp4_file)
                name = os.path.splitext(os.path.basename(mp4_file))[0]
                save_path = os.path.join('audios', name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                audio_path = os.path.join(save_path, f'{name}.wav')
                with open('audio.txt', 'a') as file:
                    file.write(audio_path+'\n')
                audio.write_audiofile(audio_path)
    elif file_path.endswith(".m3u8"):
        name, _ = os.path.splitext(file_path)
        out_name = name + ".mp4"
        command = [
        'ffmpeg',
        '-protocol_whitelist', 'file,http,https,tcp,tls,crypto',
        '-i', file_path,
        '-c', 'copy',
        '-bsf:a', 'aac_adtstoasc',
        out_name
    ]
        #ffmpeg -protocol_whitelist file,http,https,tcp,tls,crypto -i playlist.m3u8 -c copy -bsf:a  aac_adtstoasc playlist.mp4

        subprocess.run(command)
        search_pattern = os.path.join("pre_mp4", name + "*.mp4")
        matching_files = glob.glob(search_pattern)
        for mp4_file in matching_files:
                audio = mp.AudioFileClip(mp4_file)
                name = os.path.splitext(os.path.basename(mp4_file))[0]
                save_path = os.path.join('audios', name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                audio_path = os.path.join(save_path, f'{name}.wav')
                audio.write_audiofile(audio_path)
                
                with open('audio.txt', 'a') as file:
                    file.write(audio_path+'\n')

                transcript = transcriber.transcribe(audio_path)
        
                save_path = os.path.join('subtitles')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    subtitle_path = os.path.join(save_path, f'{name}.txt')
                    with open('texts.txt', 'a') as file:
                        file.write(subtitle_path+'\n')
                    file_path = str(subtitle_path)
                    if transcript.text is not None:
                        with open(subtitle_path, 'w', encoding='utf-8') as file:
                            file.write(transcript.text)


        
        
    else:
            audio = mp.AudioFileClip(file_path)
            name = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join('audios', name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            audio_path = os.path.join(save_path, f'{name}.wav')
            audio.write_audiofile(audio_path)
            with open('audio.txt', 'a') as file:
                file.write(audio_path+'\n')
            transcript = transcriber.transcribe(audio_path)
        
            save_path = os.path.join('subtitles')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                subtitle_path = os.path.join(save_path, f'{name}.txt')
                            
                file_paths = subtitle_path
                with open('texts.txt', 'a') as file:
                    file.write(subtitle_path+'\n')
                if transcript.text is not None:
                    with open(subtitle_path, 'w', encoding='utf-8') as file:
                        file.write(transcript.text)

    return "Extraction complete."



@app.route('/textscrapper', methods=['GET', 'POST'])
def test_eight_components():
    if request.method == 'POST':
        try:
            name = request.form.get('movie_name')
            driver = webdriver.Chrome()
            search_url = "https://subscene.com/subtitles"
            driver.get(search_url)
            element = driver.find_element(by=By.ID, value="query")
            element.send_keys(name)
            element.submit()
            gets = driver.find_element(by=By.CLASS_NAME, value="title")
            get_the_download = gets.find_element(by=By.TAG_NAME, value="a")
            get_the_download.click()
            table = driver.find_element(by=By.TAG_NAME, value="table")
            tbody = table.find_element(by=By.TAG_NAME, value="tbody")
            get_the_download_languages = tbody.find_elements(by=By.TAG_NAME, value="a")

            for get_the_download in get_the_download_languages:
                english_subtitle_urls.append(get_the_download.get_attribute("href"))
                for url in english_subtitle_urls:
                    if re.search(r'english', url, re.IGNORECASE):
                        english_downloads.append(url)

            # Write URLs to a text file
            with open("subtitle_urls.txt", "a") as file:
                for url in english_downloads:
                    file.write(url + "\n")
                return "Success"
            return "Success"
        except:
            return "Failure"
        
    else:
        return "Failure"


@app.route('/downloadtext',methods=['GET','POST'])
def downloads_Scrapper():
      if request.method == 'POST':
        try:
            driver = webdriver.Chrome()

# Read the first URL from the subtitle_urls.txt file
            with open("subtitle_urls.txt", "r") as file:
                    first_url = file.readline().strip()

            for user_agent in user_agent_array:
                options = webdriver.ChromeOptions()
                options.add_argument(f'user-agent={user_agent}')
                driver = webdriver.Chrome(options=options)
                driver.get(first_url)
    
    # Wait for the page to load
              
    
                header = driver.find_element(By.CLASS_NAME, "header")
                element = driver.find_element(By.CLASS_NAME, "download")
                elements_btn = element.find_element(By.TAG_NAME, "a")
                webbrowser.open(str(elements_btn.get_attribute("href")))
    
    
    # Add any additional actions you want to perform on the page here
    
    # Close the current tab
            driver.close()
    
# Quit the browser
            driver.quit()
            return "Success"
        except:
            return "Failure"
        else:
            return "Failure"

@app.route("/videoscrapper", methods=["GET", "POST"])
def videoscrapper():
    if request.method == "POST":
        names = request.form.get('movies_name')   
        ia = imdb.IMDb()
        search = ia.search_movie(names)
        for i in range(len(search)):
            movie_id = search[i].movieID
            print(search[i]['title'] + " : " + movie_id)     
            url = "https://www.2embed.to/embed/imdb/movie?id={}".format(movie_id)
            response = requests.get(url)
            webbrowser.open(url)
        if response.status_code == 404:
            print("Page not found. Closing browser.")
          
        else:
            # Open the URL in the default web browser
            webbrowser.open(url)
    
    return "Success. Yayy!!"
  

@app.route('/audioplagarism', methods=['GET', 'POST'])
def AudioSimilarity():
    if request.method == 'POST':
        # Define the frame duration in seconds
        audio_file = request.files.get('audio_pl')
        audio_file.save('uploaded_file.mp3')
        subprocess.run(['python', 'audio.py'], bufsize=0)

        # Read the result from the JSON file
        with open("results.json", 'r') as json_file:
            result = json.load(json_file)

        # Return the result as JSON
        return jsonify(result)

    return render_template('index.html')
    
@app.route('/videosimilarity',methods=['GET','POST'])
def videoSimilarity():
    global file_paths
    global video_file_path
    if request.method == "POST":
        file = request.files.get('video_filename')
        if file.filename.endswith(".mkv"):
            if not os.path.exists("pre_mp4"):
                os.mkdir("pre_mp4")
            file_path = os.path.join("pre_mp4", file.filename)
            file.save(file_path)
        else:
            if not os.path.exists("uploads"):
                os.mkdir("uploads")
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            video_file_path=file_path
            with open('vid.txt','a') as file:
                file.write(video_file_path+'\n')
     
    if file_path.endswith((".mp4", ".mov")):
        name, _ = os.path.splitext(file_path)
        audio = mp.AudioFileClip(file_path)
        name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join('audios', name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        audio_path = os.path.join(save_path, f'{name}.wav')
        with open('audio.txt','a') as file:
            file.write(audio_path+'\n')
        audio.write_audiofile(audio_path)
        transcript = transcriber.transcribe(audio_path)
        
        save_path = os.path.join('subtitles')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        subtitle_path = os.path.join(save_path, f'{name}.txt')
        with open('texts.txt','a') as file:
            file.write(subtitle_path+'\n')
        file_paths = subtitle_path
        if transcript.text is not None:
            with open(subtitle_path, 'w', encoding='utf-8') as file:
                file.write(transcript.text)

    # Convert MKV to MP4
    if file_path.endswith(".mkv"):
        name, _ = os.path.splitext(file_path)
        out_name = name + ".mp4"
        print("Converting from MKV to MP4...")
    
        # Use subprocess to run the ffmpeg command
        subprocess.run(['ffmpeg', '-i', file_path, out_name])
        search_pattern = os.path.join("pre_mp4", name + "*.mp4")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            # Found MP4 file(s) with the same name
            for mp4_file in matching_files:
                audio = mp.AudioFileClip(mp4_file)
                name = os.path.splitext(os.path.basename(mp4_file))[0]
                save_path = os.path.join('audios', name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                audio_path = os.path.join(save_path, f'{name}.wav')
                with open('audio.txt', 'a') as file:
                    file.write(audio_path+'\n')
                audio.write_audiofile(audio_path)
    elif file_path.endswith(".m3u8"):
        name, _ = os.path.splitext(file_path)
        out_name = name + ".mp4"
        command = [
        'ffmpeg',
        '-protocol_whitelist', 'file,http,https,tcp,tls,crypto',
        '-i', file_path,
        '-c', 'copy',
        '-bsf:a', 'aac_adtstoasc',
        out_name
    ]
        #ffmpeg -protocol_whitelist file,http,https,tcp,tls,crypto -i playlist.m3u8 -c copy -bsf:a  aac_adtstoasc playlist.mp4

        subprocess.run(command)
        search_pattern = os.path.join("pre_mp4", name + "*.mp4")
        matching_files = glob.glob(search_pattern)
        for mp4_file in matching_files:
                audio = mp.AudioFileClip(mp4_file)
                name = os.path.splitext(os.path.basename(mp4_file))[0]
                save_path = os.path.join('audios', name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                audio_path = os.path.join(save_path, f'{name}.wav')
                audio.write_audiofile(audio_path)
                
                with open('audio.txt', 'a') as file:
                    file.write(audio_path+'\n')

                transcript = transcriber.transcribe(audio_path)
        
                save_path = os.path.join('subtitles')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    subtitle_path = os.path.join(save_path, f'{name}.txt')
                    with open('texts.txt', 'a') as file:
                        file.write(subtitle_path+'\n')
                    file_path = str(subtitle_path)
                    if transcript.text is not None:
                        with open(subtitle_path, 'w', encoding='utf-8') as file:
                            file.write(transcript.text)


        
        
    else:
            audio = mp.AudioFileClip(file_path)
            name = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join('audios', name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            audio_path = os.path.join(save_path, f'{name}.wav')
            audio.write_audiofile(audio_path)
            with open('audio.txt', 'w') as file:
                file.write(audio_path)
            transcript = transcriber.transcribe(audio_path)
        
            save_path = os.path.join('subtitles')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                subtitle_path = os.path.join(save_path, f'{name}.txt')
                            
                file_paths = subtitle_path
                with open('texts.txt', 'a') as file:
                    file.write(subtitle_path+'\n')
                if transcript.text is not None:
                    with open(subtitle_path, 'w', encoding='utf-8') as file:
                        file.write(transcript.text)
    subprocess.run(["python","tester.py"],bufsize=0)
    subprocess.run(["python","audioss.py"],bufsize=0)
    subprocess.run(["python","text.py"],bufsize=0)
    scores = []
    with open('similar.txt', 'r') as file:
        for line in file:
            scores.append(float(line.strip()))

    average = sum(scores) / len(scores)
    file_content = int(average) * 100
    return  render_template('index.html',file_content=file_content)


@app.route('/plagarismtext',methods=['GET','POST'])
def textplagarsim():
    if request.method == 'POST':
        file = request.files.get('file')
        if not os.path.exists("TextFilePlagarism"):
            os.mkdir("TextFilePlagarism")
        file_path = os.path.join("TextFilePlagarism", file.filename)
        file.save(file_path)
        with open(file_path, 'r') as file:
            srt_text = file.read()
        cleaned_text = remove_punctuation_and_numbers(srt_text)
        if FILE_PATH is not None:

            similarity_score = calculate_similarity_score(cleaned_text, file_paths )
        
        return f"Similarity Score: {similarity_score}"
    else:
        return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True, port=8080)