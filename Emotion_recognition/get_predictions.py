import tensorflow as tf
import pandas as pd
import sys
import os
import cv2


# speicherorte fuer trainierten graph und labels in train.sh festlegen ##

def predict(image_data, sess, softmax_tensor):
    # holt labels aus file in array
    label_lines = [line.rstrip() for line in tf.gfile.GFile("tf_files/retrained_labels.txt")]
    predictions = sess.run(softmax_tensor, \
              {'DecodeJpeg/contents:0': image_data})
    # gibt prediction values in array zuerueck:

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    # sortierung; circle -> 0, plus -> 1, square -> 2, triangle -> 3; array return bsp [3 1 2 0] -> sortiert nach groesster uebereinstimmmung

    # output
    emotion = label_lines[top_k[0]]
    score = predictions[0][top_k[0]]
    #print('%s (score = %.5f)' % (emotion, score))

    return (emotion, score)

def main():
    output = pd.DataFrame(columns = ['Frame_ID', 'Emotion'])
    df = pd.read_csv('../Face_detection/Test_annotated.csv')
    for image_id, img_name in enumerate(df.Frame_ID.unique()):
        image_df = df[df.Frame_ID == img_name]
        image = cv2.imread('../Data/Test/' + img_name)
        max_score = 0
        for _, row in image_df.iterrows():
            face = image[int(row['y_min']):int(row['y_max']), int(row['x_min']):int(row['x_max'])]
            success, encoded_face = cv2.imencode('.png', face)
            byte_array = encoded_face.tobytes()
             # Disable tensorflow compilation warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

            # !! labels befinden sich jeweils in eigenen lines -> keine aenderung in retrain.py noetig -> falsche darstellung im windows editor !!

            # graph einlesen, wurde in train.sh -> call retrain.py trainiert
            with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:

                graph_def = tf.GraphDef()	## The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
                graph_def.ParseFromString(f.read())	#Parse serialized protocol buffer data into variable
                _ = tf.import_graph_def(graph_def, name='')	# import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor

              #https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception.py ; ab zeile 276

            with tf.Session() as sess:

                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
              # return: Tensor("final_result:0", shape=(?, 4), dtype=float32); stringname definiert in retrain.py, zeile 1064

                emotion, score = predict(byte_array, sess, softmax_tensor)
            if score > max_score:
              max_score = score
              emotion_class = emotion
        output.loc[len(output)]=[img_name, emotion_class]
        print(img_name, emotion_class)
    output.to_csv('Test.csv', header=True, index=None)

if __name__ == "__main__":
    main()
