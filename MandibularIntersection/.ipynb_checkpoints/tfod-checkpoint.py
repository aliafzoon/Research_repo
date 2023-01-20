### You will need this in your console only if on linux
#'cd /home/vampire/env39/Tensorflow/models/research/'
#'export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim'

import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import pathlib
import csv
import numpy as np
from PIL import Image
        


class Tfod_User:
    def __init__(self,config_path, ckpt_path, label_path, img_list, pref_size, csv_name):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.img_list = img_list
        self.label_path = label_path
        self.pref_size = pref_size
        self.csv_name = csv_name
    
# Loads the model from checkpoint and returns it
    def model_loader(self):
        configs = config_util.get_configs_from_pipeline_file(self.config_path)
        self.detection_model = model_builder.build(model_config=configs["model"], is_training=False)
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(self.ckpt_path).expect_partial()

# Uses RGB images 
# Inputs list of image addresses, label map path and uniform image size
# Outputs tensor and numpy array containing RGB images
    def data_prep(self, image_list):
        category_index = label_map_util.create_category_index_from_labelmap(self.label_path)
        image_np = np.zeros((len(image_list), self.pref_size[0], self.pref_size[1], 3))
        for it in range(len(image_list)):
            im = Image.open(image_list[it])
            im = im.resize((self.pref_size), resample=Image.BICUBIC)
            image_np[it] = np.array(im.convert("RGB"), dtype=np.float32)
    
        dataset = tf.convert_to_tensor(image_np, dtype=tf.float32)
        return dataset, image_np

# Uses the passed model to predict. Outputs a dictionary.
    def detector_func(self, image_dataset):
        images, shapes = self.detection_model.preprocess(image_dataset)
        prediction_dict = self.detection_model.predict(images, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

# Inputs detection dictionary, list of image addresses and a csv adress
# Outputs the bboxes (in range 0-1) to the csv along with the corresponding image name
    def box_save(self, detections, image_list):
        header = ["file_name", "ymin", "xmin", "ymax", "xmax"]
        names =[]
        cords =[]
        for i in range(len(image_list)):
            names.append(image_list[i].parts[-1])
            cords.append(detections["detection_boxes"][i,0].numpy())

        if not os.path.exists(self.csv_name):
            with open(self.csv_name, "w", encoding="UTF8", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        with open(self.csv_name, "a+", encoding="UTF8", newline='') as f:
            writer = csv.writer(f)
            writer.writerows([p, q[0], q[1], q[2], q[3]] for p, q in zip(names, cords))


# Does the prediction using detector_func and data_prep and box_save. Batches the data in groups of 5.
# inputs list of image addsresses, label map, uniform image size, csv address and the model
    def detection_batched(self):
        n_files = len(self.img_list)
        i = 0
        while i < n_files:
            print(f"Detection of {i} files done, {n_files - i} remaining.")
            if n_files-i > 5:
                temp_image_list = self.img_list[i:i+5]
                dataset, image_np = self.data_prep(temp_image_list)
                detections = self.detector_func(dataset)
                self.box_save(detections, temp_image_list)
                i = i+5
            if n_files-i <= 5:
                temp_image_list = self.img_list[i:n_files]
                dataset, image_np = self.data_prep(temp_image_list)
                detections = self.detector_func(dataset)
                self.box_save(detections, temp_image_list)
                i = n_files

        print(f"BBox Detection of {n_files} files finished.")
        
    def cleaner(self):
        tf.keras.backend.clear_session()
        del self.detection_model

