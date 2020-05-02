# All library imports

import pandas as pd
import os
import cv2
import xml.etree.ElementTree as ET      # for parsing xml files

def get_bboxes(file_path, row):
	global annotated_data
	tree = ET.parse(file_path)
	root = tree.getroot()
	size = root.find('size')
	width = size.find('width').text
	height = size.find('height').text
	for country in root.findall('object'):
		class_name = country.find('name').text
		for bbox in country.findall('bndbox'):
			x_min = int(bbox.find('xmin').text)
			y_min = int(bbox.find('ymin').text)
			x_max = int(bbox.find('xmax').text)
			y_max = int(bbox.find('ymax').text)
			annotated_data.loc[len(annotated_data)]=[row['Frame_ID'], row['Emotion'], width, height, x_min, y_min, x_max, y_max, class_name]
	return

def save_annotation(file_path):     # for visualizing the annotation on image
	df = pd.read_csv("Test_annotated.csv")
	for _, row in df.iterrows():
		image_path = os.path.join(file_path, row['Frame_ID'])
		image = cv2.imread(image_path)
		cv2.rectangle(image, (int(round(row['x_min'])), int(round(row['y_min']))), (int(round(row['x_max'])), int(round(row['y_max']))), (0, 255, 0), 2)
		save_path = os.path.join('../Data/test_annotated', row['Frame_ID'])
		cv2.imwrite(save_path, image)


def main():
	global annotated_data
	data = pd.read_csv("../Data/Train.csv")
	annotated_data = pd.DataFrame(columns=['Frame_ID', 'Emotion', 'width', 'height', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name'])

	for index, row in data.iterrows():
		try:
			xml_file = row['Frame_ID'].split('.')[0]+'.xml'
			print(xml_file)
			if os.path.isfile('Train_xml/'+ xml_file):
				bbox = get_bboxes('Train_xml/'+ xml_file, row)
				print(annotated_data)
		except Exception as e:
			print(e)
			pass
	annotated_data.to_csv('Train_annotated.csv', header=True, index=None)

if __name__ == "__main__":
	#main()
	save_annotation('../Data/test_frames')
