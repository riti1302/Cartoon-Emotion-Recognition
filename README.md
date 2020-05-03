# Cartoon-Emotion-Recognition

### Prerequisites

This Repository is maintained on Python 3.7 version.

  - CPython  3.6.9
  - numpy  1.18.2
  - pandas  1.0.3
  - torch  1.4.0
  - torchvision  0.5.0
  - detectron2  0.1.1
  - tensorflow  1.15.0


### Dataset

Dataset consists of a [training](Data) video and [testing](Data) video along with csv file which contains the emotions of the cartoon corresponding to each frame (frame rate = 5) of the video.

### Preprocessing

Dataset is in the form of video file from which frames are needed to be extracted. Frames from the video can be obtained using `get_frames.py`.           
The faces for face detection are annotated using [labelImg](https://github.com/tzutalin/labelImg). The annotations for each frame can be transferred to a csv file using `xml_to_csv.py`.  
   
Dataset for emotion recognition can be obtained using `get_dataset.py`. It takes a csv file containing the coordinates for the faces and its corresponding emotion and create a dataset for the emotion recognition model.   
If you want to train the emotion recognition model on a custom dataset then keep the dataset inside  [training_dataset](Emotion_Recognition/training_dataset) images of emotion in a saperate folder.

### Training 
#### Face Detection model

The training of face detection model is done in `Cartoon_Face_Detection.ipynb`. The trained weights can be downloaded from [here](something). Function for prediction on a new image is also present in this notebook.

#### Emotion Recognition Model
Run the `train.sh` file to start the training.    
Prediction on a new image can be done using `get_prediction.py`  


### Outputs

- <p align="center"> <img src="238.jpg"/> </p>


