import os
import sys

import cv2
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.visualize import display_instances

from tkinter import filedialog
from tkinter import *

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "./",title = "Select Image",
                                            filetypes = (
                                                ("jpeg files","*.jpg"),
                                                ("all files","*.*"))
                                            )
class MydataConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "mydata"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + mydata

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

config = MydataConfig()
config.display()
model_dir = "../../"
model_path = "../../logs/mydata20200813T1923/mask_rcnn_mydata_0012.h5"
#model_path = "../../logs/mydata20200810T0753/mask_rcnn_mydata_0003.h5"
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
model.load_weights(model_path,by_name=True)
class_name = ['BG','Elephent']

img = cv2.imread(root.filename)
dsize = (400,400)

results = model.detect([img],verbose=1)

r = results[0]
classes= r['class_ids']
print("Total Objects found", len(classes))
for i in range(len(classes)):
    print(class_name[classes[i]])

#img = cv2.resize(img,dsize)
cv2.imshow("test",img)
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_name, r['scores'])


