from pydub import AudioSegment
import io
import requests
import os
import json

frame_duration = 10
file_path = os.getcwd()

try:
    # Open the audio file using pydub
    audio = AudioSegment.from_file(os.path.join(file_path, "uploaded_file.mp3"))

    # Calculate the start and end times of the frame
    start_time = 0
    end_time = frame_duration * 1000

    # Extract the frame from the audio using pydub
    frame = audio[start_time:end_time]

    # Export the frame as an mp3 file
    frame.export("frame.mp3", format="mp3")

    # Read the frame audio file as binary data
    with open("frame.mp3", 'rb') as file:
        audio_data = file.read()

    # API endpoint for music recognition
    url = "https://api.audd.io/"

    # Parameters for the API request
    params = {
        'api_token': "<YOUR API KEY>",
        'return': 'timecode',
    }

    # Send the API request with the frame audio data
    response = requests.post(url, params=params, files={'file': io.BytesIO(audio_data)})

    # Process the response
    if response.status_code == 200:
        result = response.json()
        with open("results.json","w") as json_file:
            json.dump(result,json_file)
        print("success")

        # Extract the recognized song details from the response
        # and do further processing as needed
        print(result)
    else:
        print('Error:', response.status_code)
        print(response.text)

except Exception as e:
    print('An error occurred:', str(e))
