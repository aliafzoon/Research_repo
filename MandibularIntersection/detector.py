from PIL import Image
import numpy as np
import pathlib, random, os
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LSTM, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Dropout, Activation, MaxPooling2D, Reshape


class Model_Handler:
    def __init__(self, model_path, img_adrs_list, csv_name, img_size, img_side, batch_size):
        self.model_path = model_path
        self.img_adrs_list = img_adrs_list
        self.csv_name = csv_name
        self.img_size = img_size # preffered size of (y, x)
        self.img_side = img_side
        self.batch_size = batch_size
        
    
    def load_worker(self, img_size, file_list):
        im_y, im_x = img_size
        n_files = len(file_list)
        f_array = np.zeros((n_files,im_x, im_y))                       #grayscale

        for i in range(n_files):
            image_load = Image.open(file_list[i])
            if image_load.size != img_size:
                image_load = image_load.resize(img_size)
            if len(image_load.mode) >=2:
                image_load = ImageOps.grayscale(image_load)
            if self.img_side == "right":
                # make it a left side image
                image_load = image_load.transpose(Image.FLIP_LEFT_RIGHT)
            f_array[i] = np.array(image_load)                

        f_array = np.expand_dims(f_array, axis=3)                       #grayscale
        return f_array
    
    def predictor(self, file_array):
        self.imgs_dataset = tf.data.Dataset.from_tensor_slices(file_array).batch(self.batch_size)
        model = tf.keras.models.load_model(self.model_path)
        preds = model.predict(self.imgs_dataset)
        return preds
    
    def csv_saver(self, addrs, pred_data, save_adrs):
        # pred_data in form of (file address, pred value)
        headers = ["file address", "Predicted Class", "Predicted Contact Chance"]
        pred_df = pd.DataFrame(columns=headers)
        row = 0
        print(pred_data)
        for i, j in zip(addrs, pred_data):
            addrs = str(i.parts[-1])
            pred_cls = "Contact" if j[0] > 0.5 else "Non-Contact"
            pred_perc = j[0] * 100
            pred_df.loc[row] = [addrs, pred_cls, f"{pred_perc:0.3f}%"]
            row = row + 1
        pred_df.to_csv(save_adrs)
            
    def auto_handler(self):
        file_array = self.load_worker(self.img_size, self.img_adrs_list)
        preds = self.predictor(file_array)
        self.csv_saver(self.img_adrs_list, preds, self.csv_name)