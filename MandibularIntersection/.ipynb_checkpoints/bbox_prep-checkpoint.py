from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import pathlib
import os
import pandas


class Bbox_User:
    def __init__(self, im_list, bbox_path, main_image_path, save_path, factors, prefix="c", size=(220, 220)):
        self.im_list = im_list
        self.bbox_path = bbox_path
        self.main_image_path = main_image_path
        self.save_path = save_path
        self.factors = factors
        self.prefix= prefix
        self.size = size
    
    def bbox_csv_tp_list(self):
        bbox_df = pandas.read_csv(self.bbox_path)
        self.bbox_list = []                     # list of coordinates [ymin, xmin, ymax, xmax]
        self.im_names = []                      # list of image names with same order as bboxes
        for index, row in bbox_df.iterrows():
            self.bbox_list.append(list(row.iloc[1:5]))
            self.im_names.append(row["file_name"])

    def extractor(self):
        """
            Inputs a list of paths for png images and list of 4 elemen lists for boxes in [ymin, xmin, ymax, xmax] format
            Depending on the image being right or left side of the jaw, shifts to include the canal.
            Removes the Crow of the teeth
            Outputs list of numpy arrays 
        """    
        extracted = []
        percent_to_shift = 0.3  
        for im, box in zip(self.im_names, self.bbox_list):
            im_side = "left" if "left" in im else "right"
            im = np.array(Image.open(self.main_image_path / im))
            im_y, im_x = im.shape[0], im.shape[1]
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin*im_y), int(xmin*im_x), int(ymax*im_y), int(xmax*im_x)
            shift_y = int((ymax-ymin) * percent_to_shift)
            shift_x = int((xmax-xmin) * percent_to_shift)
            if im_side == "left":
                box_x = [xmin-int(shift_x*1.4), xmax-shift_x]
                box_y = [ymin+shift_y, ymax+int(shift_y*1.4)]
            else:
                box_x = [xmin+shift_x, xmax+int(shift_x*1.4)]
                box_y = [ymin+shift_y, ymax+int(shift_y*1.4)] 
            im_crop = im[box_y[0]:box_y[1], box_x[0]:box_x[1]]
            extracted.append(im_crop)
        return extracted

    def img_saver(self, img_array_list, img_names, save_path):
        for row in range(len(img_array_list)):
            to_save = Image.fromarray(img_array_list[row][:,:])
            try:
                if img_array_list[row].shape[2]:
                    to_save = ImageOps.grayscale(to_save)
            except:
                nothint = "it's grayscale"
            to_save.save(str(save_path) + f"\\{self.prefix}" + str(img_names[row]))
        print(f"{len(img_array_list)} Files saved in {save_path}")

    ### changes contrast and resizes
    def resizer(self, img_array_list):
        # Inputs list of arrays, resizes and returns list of arrays
        resized = []
        for im in img_array_list:
            temp = Image.fromarray(im).resize(self.size, resample=Image.BICUBIC)
            resized.append(np.array(temp))
        return resized

    def contrast_changer(self, img_array_list, factor):
        # Inputs list of arrays, changes contrast by factor and returns list of arrays
        contrasted = []
        for im in img_array_list:
            enhanced = ImageEnhance.Contrast(Image.fromarray(im))
            contrasted.append(np.array(enhanced.enhance(factor)))
        return contrasted
    
    def auto_handler(self):
        self.bbox_csv_tp_list()
        extracted = self.extractor()
        resized = self.resizer(extracted)
        orig_folder = self.save_path / "original"
        if not os.path.isdir(orig_folder):
            os.mkdir(orig_folder)
        self.img_saver(resized, self.im_names, orig_folder)
        
        for fac in self.factors:
            fac_folder = self.save_path / f"contrast{fac}"
            if not os.path.isdir(fac_folder):
                os.mkdir(fac_folder)
            cont_changed = self.contrast_changer(resized, fac)
            self.img_saver(cont_changed, self.im_names, fac_folder)



