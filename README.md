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

