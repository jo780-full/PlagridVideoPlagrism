# Plagrid

PlayGrid is a Python-based web application that offers various functionalities related to video and audio processing. It leverages popular libraries such as Flask, MoviePy, AssemblyAI, IMDbPY, Selenium, and more. Below, you will find an overview of the main features and how to run the application.

## Features

1. Video and Audio Extraction: The application allows users to upload video files (.mp4, .mov, .mkv) and extract the audio, which is then saved in .wav format. The extracted audio is used for further processing.

2. Automatic Subtitle Generation: PlayGrid utilizes the AssemblyAI library to automatically generate subtitles from the extracted audio. The subtitles are saved in .txt format for each uploaded video.

3. Text-based Plagiarism Detection: Users can upload text files, and the application performs text-based plagiarism detection by calculating the similarity score between the uploaded text and the generated subtitles for each video.

4. IMDb Movie Search: The app integrates IMDbPY to search for movies on IMDb based on user input. It opens the movie's page in a web browser.

5. Video Similarity Detection: PlayGrid allows users to upload multiple videos (.mp4, .mov, .mkv) and calculates the similarity score between the extracted features of each pair of videos. The similarity scores are displayed as a percentage on the user interface.

6. Web Scraping for Subtitles: The application uses Selenium to scrape subtitles for movies from the Subscene website. Users can input the movie name, and the application will attempt to fetch the English subtitles for that movie.

## How to Run the Application

To run the PlayGrid application, follow these steps:

1. Clone the repository: `git clone https://github.com/your_username/playgrid.git`

2. Change to the project directory: `cd playgrid`

3. Install the required libraries: `pip install -r requirements.txt`

4. Set up the environment variables:
   - Create a file named `.env` in the project root directory.
   - Add the following line to the `.env` file, replacing `YOUR_ASSEMBLYAI_API_KEY` with your actual AssemblyAI API key:

   ```
   ASSEMBLYAI_API_KEY=YOUR_ASSEMBLYAI_API_KEY
   ```

5. Download the spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```

6. Run the Flask web application:
   ```
   python app.py
   ```

7. Access the application in your web browser at `http://127.0.0.1:8080/`.

## Usage

1. **Video and Audio Extraction**:
   - Click on the "Choose File" button under "Video and Audio Extraction."
   - Select a video file (.mp4, .mov, .mkv) to upload.
   - The audio will be extracted and saved as a .wav file in the "audios" folder.
   - Subtitles will be automatically generated and saved as a .txt file in the "subtitles" folder.

2. **Text-based Plagiarism Detection**:
   - Click on the "Choose File" button under "Text-based Plagiarism Detection."
   - Select a text file to upload.
   - The application will compare the uploaded text with the generated subtitles for each video and display the similarity score.

3. **IMDb Movie Search**:
   - Enter the movie name in the "Movie Name" field under "IMDb Movie Search."
   - Click the "Search" button, and the application will open the IMDb page for the movie in a new web browser tab.

4. **Video Similarity Detection**:
   - Click on the "Choose File" button under "Video Similarity Detection."
   - Upload multiple video files (.mp4, .mov, .mkv) to compare.
   - The application will calculate the similarity score between each pair of videos and display the average similarity score.

5. **Web Scraping for Subtitles**:
   - Enter the movie name in the "Movie Name" field under "Web Scraping for Subtitles."
   - Click the "Search Subtitles" button.
   - The application will attempt to fetch the English subtitles for the movie and open the download link in a web browser.

## Note

- The application uses the AssemblyAI API for automatic transcription of audio to generate subtitles. Make sure to replace `YOUR_ASSEMBLYAI_API_KEY` in the `.env` file with your actual AssemblyAI API key.

- The video similarity detection feature relies on pre-trained ResNet50 models to extract features from video frames. Make sure you have the required libraries and models installed.

- The IMDb movie search feature depends on the IMDbPY library to search for movies on IMDb. It requires an active internet connection to fetch movie details.

- The web scraping feature uses Selenium and requires an internet connection to access the Subscene website and fetch subtitles.

- Be cautious while using the web scraping feature to avoid violating any website's terms of service.
