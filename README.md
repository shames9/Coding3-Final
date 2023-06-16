# Realtime-emotion-detectionusing-videotest

Realtime-emotion-detectionusing-videotest [Open in Colab](https://colab.research.google.com/drive/1Jg8ZECPaOY_FmDiLabwkq413qpD8X9BJ#scrollTo=XYsz4YVnzK48)

![images](https://github.com/shames9/Coding3-Final/blob/main/images/fear.result.png)

[Presentation Video](https://youtu.be/wc7epeTrays)

## Introduction
I hope that by analysing facial expressions, we can better understand and respond to people's emotional states. The code provides a simple and practical base framework to help developers quickly build and deploy applications for facial emotion analysis. By combining computer vision and deep learning techniques, we can create a more intelligent and human interaction experience.

This code is a deep learning based facial emotion analysis application. It uses computer vision and machine learning techniques to detect faces captured by the camera and to predict the emotion category of facial expressions. The code performs face detection using OpenCV by loading pre-trained models and weights, and pre-processes and classifies the detected face regions for emotion.

## Design and Development
Because I had seen some videos on YouTuBe about the analysis of facial emotions and the analysis of facial disorders of human faces in the videos made me find this aspect interesting, I searched for a lot of databases on facial expressions and combined them with deep learning in the process of designing this project.

In the code, I first Importing necessary libraries for plotting and image processing
```ruby
import os                       #For interaction with the operating system
import cv2                       #OpenCV library for image processing
import numpy as np                  #For numerical calculations
from keras.models import model_from_json       #Functions for loading models from JSON files in Keras
from keras.preprocessing import image        #Image pre-processing tools are provided

from IPython.display import display, Javascript   #For displaying images or other content，Allow JavaScript code to be executed
from google.colab.output import eval_js       #For evaluating JavaScript code
from base64 import b64decode             #For decoding Base64 encoded data

from IPython.display import HTML, Audio       #Module for displaying HTML content and playing audio

import matplotlib.pyplot as plt           #For creating charts
from google.colab.patches import cv2_imshow     #Function from the google.colab.patches module for displaying images in Google Colab.
from google.colab.output import eval_js
from base64 import b64decode             #For decoding Base64 encoded data
import numpy as np
import io                       #For input/output operations
from PIL import Image
```

Then try capturing the video frames in the web page and converting them to images
```ruby
VIDEO_HTML = """
<video autoplay
 width=%d height=%d style='cursor: pointer;'></video>
<script>

var video = document.querySelector('video')

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream=> video.srcObject = stream)

var data = new Promise(resolve=>{
  video.onclick = ()=>{
    var canvas = document.createElement('canvas')
    var [w,h] =[video.offsetWidth, video.offsetHeight]
    canvas.width = w
    canvas.height = h
    canvas.getContext('2d')
          .drawImage(video, 0, 0, w, h)
    video.srcObject.getVideoTracks()[0].stop()
    video.replaceWith(canvas)
    resolve(canvas.toDataURL('image/jpeg', %f))
  }
})
</script>
"""
```

Then I defined a function called take_photo to capture a photo from the video and return its image data
```ruby
def take_photo(filename='photo.jpg', quality=0.8, size=(800,600)):  #A function called take_photo is defined which accepts as arguments the filename, image quality and size, with default values of 'photo.jpg', 0.8 and (800,600) respectively.
  display(HTML(VIDEO_HTML % (size[0],size[1],quality)))       #Display the HTML content in a Jupyter/Colab notebook, using the previously defined string VIDEO_HTML and inserting the values of size and quality into the HTML.
  data = eval_js("data")                       #Get the value of the data variable defined earlier by executing the JavaScript code eval_js("data").
  binary = b64decode(data.split(',')[1])               #Separate the values of the data variable by commas and decode the Base64 encoded image data.
  f = io.BytesIO(binary)                       #Store the decoded image data in the io.BytesIO object.
  return np.asarray(Image.open(f))                  #Open the image in f and return it after converting it to a NumPy array.
                                     #The function does this by displaying the video stream and waiting for the user to click to capture a photo, then converting the photo to a NumPy array to be returned for further processing or analysis.
```

The following code is the core part of this project and implements the following functions:

1. Open the default camera device and create a VideoCapture object and assign it to the cap variable.
2. Enter an infinite loop for continuous processing of video frames.
3. Call the take_photo function to capture a picture and assign it to the test_img variable.
4. Convert the captured colour image into a grey-scale image for subsequent face detection.
5. Use the face_haar_cascade cascade classifier object to detect the face in the grey-scale image, returning the coordinates of the rectangular region of the face.
6. For each face detected, a blue rectangular box is drawn and the region of interest (ROI), the face region, is truncated. The ROI is then resized to 48x48 and the data is normalised.
7. Use the pre-trained model mod to predict the sentiment of the normalised face images
8. Find the emotion category with the highest probability in the prediction result and convert it to the corresponding emotion label.
9. Draw the predicted emotion labels on the image.
10. Resize the image and use Matplotlib to plot the image and display the predicted results.
```ruby
cap=cv2.VideoCapture(0)                              #Open the default camera device and create a VideoCapture object and assign it to the cap variable.

while True:                                    #Enter an infinite loop for continuous processing of video frames
    test_img=take_photo()                           # captures frame and returns boolean value and captured image

    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)           #Converts the captured colour image into a greyscale image for subsequent face detection.



    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5) #Use the face_haar_cascade cascade classifier object to detect faces in a grey-scale image, returning the coordinates of the rectangular region of the face.


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]                   #cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))

        img_pixels = roi_gray.astype('float32')
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0
                                        #For each face detected, a blue rectangular box is drawn and the region of interest (ROI), the face region, is truncated. The ROI is then resized to 48x48 and the data is normalised.

        predictions = model.predict(img_pixels)             #Emotion prediction on normalised face images using a pre-trained model mod.

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
                                        #Find the sentiment category with the highest probability in the predicted outcome and convert it to the corresponding sentiment label.
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) #Plotting the predicted sentiment labels on the image.

    resized_img = cv2.resize(test_img, (1000, 700))
                                        #cv2.imshow('Facial emotion analysis ',resized_img)
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
                                        # as opencv loads in BGR format by default, we want to show it in RGB.
    plt.show()

    if cv2.waitKey(10) == ord('q'):                   #wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
#Free up camera resources and close all windows.
```

## Summary and Reflection
In this project, I have implemented a simple and practical facial emotion analysis application through a python design and development process utilising tools and techniques such as OpenCV, Keras and deep learning models. Using a live video stream captured by a camera, it was able to detect faces and predict emotion categories, tagging the predictions to be displayed on the image. Unfortunately, in the project I would have liked to include an audio feedback feature for emotion recognition and UI interface elements to make the final detected emotion probabilities more visual. But it seems that there is more than just that functionality in cloab, and if I had more time I would have put python in vscode to refine the research and complete the functionality I was hoping for.

In future projects, I can further optimise the model and interface to achieve a more accurate and personalised facial emotion analysis application based on my needs and dataset. Improve the training of a database myself (the emotion dataset on kaggle is too small) to make the detected results more accurate.

## Reference
https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

https://morioh.com/p/801c509dda99

https://github.com/Dhanush45/Realtime-emotion-detectionusing-python

