import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')
target_size = (224, 224)

def predict(model, img, target_size, top_n=3):
 """Run model prediction on image
 """
 if img.size != target_size:
   img = img.resize(target_size)
 x = image.img_to_array(img)
 x = np.expand_dims(x, axis=0)
 x = preprocess_input(x)
 preds = model.predict(x)
 return decode_predictions(preds, top=top_n)[0]

def plot_preds(image, preds):
 """Displays image and the 
 top-n predicted probabilities in a bar graph
 """
 plt.imshow(image)
 plt.axis('off')
 plt.figure()
 order = list(reversed(range(len(preds))))
 bar_preds = [pr[2] for pr in preds]
 labels = (pr[1] for pr in preds)
 plt.barh(order, bar_preds, alpha=0.5)
 plt.yticks(order, labels)
 plt.xlabel('Probability')
 plt.xlim(0,1.01)
 plt.tight_layout()
 plt.show()
  
response = requests.get("http://bit.ly/2jUqlmu")
img = Image.open(BytesIO(response.content))
preds = predict(model, img, target_size)
plot_preds(img, preds)
