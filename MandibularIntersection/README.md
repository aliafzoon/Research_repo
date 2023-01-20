
# Mandibular Intersection Detector 

This repository contains Tensorflow implementation of the following paper: ***
In this paper, we detect the contact between the third mandibular molar root and the mandibular canal.  
We use two separate networks to localize and classify the tooth. A fine tuned Tensorflow Object Detection model of [EfficientDet D1](https://arxiv.org/abs/1911.09070)  and an optimized CNN model.


This repo contains IPython notebooks. 
## Installing Prerequisites

To use these files, a minimum Python version of 3.9 is required.
To use these files properly, you need to install a Jupyter notebook or
Jupyter lab.
```bash
  pip install jupyterlab
```
To install the Tensorflow Object Detection api, use [this](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#) tutorial. 

To install the extra packages:
```bash
  pip install numpy pandas pathlib pillow tensorflow 
```

## Usage

You need to setup TFOD api first. Download and set up the Efficientdet-D1 model. Put the "workspace"  folder inside your "tensorflow" folder of TFOD and replace the "pipeline.config".
The next step is to use the cropper.py to separate the panoramic x-rays into a left-hand side and a right-hand side images. Next, Run the "Master_user.ipynb" and adjust the paths according to your directory. After using the Detection model, go to the created images directory and choose the best contrast level created. Contrast factor of 1.5 is recommended. Separate the right-hand side and left-hand side images into independent folders and enter their paths in "Master_user.ipynb" for the classification part. Run the specified cells to get a CSV file containing the classification results. 
You can get the trained weights of the classification model alongside with the dataset upon reasonable request
through aliafzoon@gmail.com.

  
## Citing Our Paper 

If you use this repository or would like to refer the paper,
 please use the following entry

...

  
## Paper Authors

- [Ali Afzoon](https://www.github.com/aliafzoon)
- [Abolfazl Shiri](addemail@email.com)
- ...

  