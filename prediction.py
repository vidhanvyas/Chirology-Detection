import os
import pandas as pd
from gtts import gTTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow logging level to suppress warnings
import cv2  # OpenCV library for computer vision
import mediapipe as mp  # Google's Mediapipe library for hand tracking
from keras.models import load_model  # Loading a pre-trained model
import numpy as np
import time

# Load the pre-trained sign language alphabet classification model
model = load_model('smnist.h5')

# Initialize the Mediapipe Hands module for hand tracking
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open the video capture stream (camera)
cap = cv2.VideoCapture(0)

# Read a frame from the video stream to get the frame dimensions
_, frame = cap.read()
h, w, c = frame.shape

# List of letters for prediction
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
              'W', 'X', 'Y']

# Main loop for capturing and processing frames
while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, Thank you maam :)")
        break
    elif k % 256 == 32:
        # SPACE pressed
        analysisframe = frame
        showframe = analysisframe
        cv2.imshow("Frame", showframe)

        # Convert frame to RGB and process hand landmarks
        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks

        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lmanalysis in handLMsanalysis.landmark:
                    x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 30
                y_max += 30
                x_min -= 30
                x_max += 30

                # Convert analysis frame to grayscale
        analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)

        # Extract ROI and resize to 28x28
        analysisframe = analysisframe[y_min:y_max, x_min:x_max]
        analysisframe = cv2.resize(analysisframe, (28, 28))

        # Flatten pixel values
        nlist = []
        rows, cols = analysisframe.shape
        for i in range(rows):
            for j in range(cols):
                k = analysisframe[i, j]
                nlist.append(k)

        # Create DataFrame and reshape pixel data
        datan = pd.DataFrame(nlist).T
        colname = []
        for val in range(784):
            colname.append(val)
        datan.columns = colname

        # Prepare pixel data for prediction
        pixeldata = datan.values
        pixeldata = pixeldata / 255
        pixeldata = pixeldata.reshape(-1, 28, 28, 1)

        # Use the model to predict class probabilities
        prediction = model.predict(pixeldata)

        # Convert prediction results to a NumPy array
        predarray = np.array(prediction[0])

        # Create a dictionary to map predicted probabilities to letters
        letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}

        # Sort predicted probabilities in descending order
        predarrayordered = sorted(predarray, reverse=True)

        # Get the highest predicted probability
        high1 = predarrayordered[0]
        high2 = predarrayordered[1]

        # Create a string to store the output
        output_text = ""

        # Display the top predicted character and its confidence
        for key, value in letter_prediction_dict.items():
            if value == high1:
                # Store the top predicted character and its confidence in the output string
                output_text += key

                # Display the top predicted character and its confidence
                print("Predicted Character 1: ", key)
                print('Confidence 1: ', 100 * value)

                # Display the second top predicted character and its confidence
            elif value == high2:
                print("Predicted Character 2: ", key)
                print('Confidence 2: ', 100 * value)

        # # Convert the output text to speech
        # tts = gTTS(text=output_text, lang="en")

        # # Save the speech to a temporary file
        # tts.save("output.mp3")

        # # Play the speech using the default media player
        # os.system("afplay output.mp3")

        # # Pause execution for 5 seconds
        # time.sleep(5)

        # # Delete the temporary file
        # os.remove("output.mp3")

    # Convert the frame to RGB format for Mediapipe processing
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the Hands module to detect hand landmarks
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 30
            y_max += 30
            x_min -= 30
            x_max += 30
            # Draw a rectangle around the detected hand region
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)

        # Display the processed frame with annotations
        cv2.imshow("Frame", frame)

# Release the camera and close all windows
cap.release() 
cv2.waitKey(1)
cv2.destroyAllWindows()
