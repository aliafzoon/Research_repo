"""
	This script devides pictures in a directory into a right side and a left side.

	Please enter your image path in the specified line. 
	An example for image path: pa = pathlib.Path("./first_xrays") 
	Please note to use "\\" instead of "/" in the address if you are on windows
"""
 
from PIL import Image
import numpy as np
import pathlib, os

pa = pathlib.Path("YOUR_PATH_HERE")

im_list = list(pa.glob('./*.png'))
base_address = pa.parents[0] / "cropped"
if not os.path.isdir(base_address):
    os.mkdir(base_address)

def splitter(list_im):
    for im in list_im:
        img = np.array(Image.open(im))
        mid = int(img.shape[1] / 2)
        imgL = Image.fromarray(img[:,:mid])
        imgR = Image.fromarray(img[:,mid:])
        imgL.save(str(base_address / im.parts[-1])[:-4] + "left.png")
        imgR.save(str(base_address / im.parts[-1])[:-4] + "right.png")


splitter(im_list)
print("Crop for {} files -> Successful".format(len(im_list)))
