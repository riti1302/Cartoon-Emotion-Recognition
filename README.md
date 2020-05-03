# Cartoon-Emotion-Recognition

### Dataset
Dataset consists of a [training](Data) video and [testing](Data) video along with csv file which contains the emotions of the cartoon corresponding to each frame (frame rate = 5) of the video.

For frames can be obtained using `get_frames.py`. The faces for face detection is annotated using [labelImg](https://github.com/tzutalin/labelImg). The annotations can be transferred to csv file using `xml_to_csv.py`.

## Training 
### Face detection model
The training of face detection model is done in `Cartoon_Face_Detection.ipynb`. The trained weights can be downloaded from [here](something). Functions for prediction on a new image is also present in this notebook.

### Emotion Recognition Model
The training of the emotion recognition model is done in `Cartoon_Emotion_Recognition.ipynb`. The trained weights can be downloaded from [here](something). Input to this is a csv file containing the coordinates of the detected face and output is a csv file containing the recognised emotion for each frame.

