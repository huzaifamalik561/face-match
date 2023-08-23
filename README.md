# Face Matching Web App

This is a Flask-based web application that performs face matching between an uploaded ID card image and a user-provided image. The app calculates the similarity between the two faces and displays the results along with extracted faces and eye distances.

## Features

- Upload an ID card image and a user image to compare.
- Extract faces from the images and display them.
- Calculate the distance between the eyes in the images.
- Determine if the images match based on the calculated similarity.

## Getting Started

Follow the steps below to set up and run the Face Matching App locally.

### Prerequisites

- Python 3.x installed on your system.
- Basic knowledge of Flask and web development.

### Installation

1. Clone the repository:
   git clone https://github.com/your-username/face-matching-app.git
   cd face-matching-app
Install the required packages:

bash
pip install -r requirements.txt
Usage
Run the application:

bash
python app.py
Open a web browser and navigate to http://127.0.0.1:5000/.

Upload an ID card image and a user image.

Click the "Match" button to perform the face matching.

View the results, extracted faces, and calculated eye distances.

Customization
You can adjust the similarity threshold and other parameters in the app.py file.
Customize the HTML/CSS templates in the templates directory for UI modifications.
Technologies Used
Flask: Micro web framework for building the web app.
OpenCV: Library for computer vision tasks.
face_recognition: Library for face recognition tasks.
MTCNN: Multi-task Cascaded Convolutional Networks for face detection.
Bootstrap: CSS framework for styling the user interface.
Acknowledgements
This app is based on the original code provided by Huzaifah Bin Khawar.
The face recognition and image processing libraries are used for the core functionality.
