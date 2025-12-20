GTSRB-ML: Traffic Sign Classifier
**What is this?**

For this project, I built a deep learning model that can recognize 43 different types of German traffic signs. I used a method called Transfer Learning with a pre-trained model (MobileNetV2), which basically means I took a model that already knows how to "see" shapes and colors and taught it specifically about traffic signs.

**Why it's cool**

_High Accuracy:_ It hit 99.3% accuracy on the validation set and 93.5% on the final test set.

_Smart Training:_ I used "Data Augmentation" to flip, zoom, and tilt the images so the model doesn't get confused by bad lighting or weird angles.

_Ready to Use: _ I wrapped the whole thing in a Gradio web app, so you can just upload a photo of a sign and it tells you what it is instantly.


**The Tech Stack**


The Brains: Python & TensorFlow/Keras.

The Math: Pandas & Numpy for handling the data.

The Visuals: Matplotlib & Seaborn for the charts, OpenCV for processing the images.

The UI: Gradio (for the interactive dashboard).

**How I structured the code**

I broke the project down into 5 main parts:

Config: Where I keep all the settings like paths and image sizes.

DataManager: This script downloads the dataset and gets the folders ready.

EDAExplorer: This is where I plotted the data to see which signs appeared the most.

TrafficSignModel: The actual training code where the MobileNetV2 magic happens.

Predictor: A simple tool to take a new image and get a prediction.

**How to run it**

I mostly built this to run on Google Colab (it’s easier because of the free GPU), but you can run it locally too.

1. Install the libraries:

Bash

pip install tensorflow pandas numpy matplotlib seaborn opencv-python scikit-learn gradio
2. Get the data: I wrote a helper script to download the GTSRB dataset for you. Just make sure you update the file paths in the Config section of the notebook to match where you saved it.

3. Launch the UI: Once the model is trained, the Gradio app will give you a link to a live interface where you can test your own images.

The Results
The Confusion Matrix (the chart that shows where the model gets confused) shows that it’s really good at most signs, only struggling slightly with very similar-looking speed limit signs.
