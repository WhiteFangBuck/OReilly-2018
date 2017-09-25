import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import datetime as dt

startTime = dt.datetime.now()

#fixed size for InceptionV3
IM_WIDTH, IM_HEIGHT = 299, 299 
NB_EPOCHS = 3
BAT_SIZE = 16
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172


def get_nb_files(directory):
 """Get number of files by 
 searching directory recursively"""
 if not os.path.exists(directory):
  return 0
 cnt = 0
 for r, dirs, files in os.walk(directory):
  for dr in dirs:
   cnt += len(glob.glob(os.path.join(r, dr + "/*")))
 return cnt


def setup_to_transfer_learn(model, base_model):
 """Freeze all layers and compile the model"""
 for layer in base_model.layers:
  layer.trainable = False
 model.compile(
               optimizer='rmsprop', 
               loss='categorical_crossentropy', 
               metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
 """Add last layer to the convnet
 """
 x = base_model.output
 x = GlobalAveragePooling2D()(x)
 x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
 predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
 model = Model(input=base_model.input, output=predictions)
 return model

def setup_to_finetune(model):
 """Freeze the bottom NB_IV3_LAYERS 
 and retrain the remaining top layers.
 note: NB_IV3_LAYERS corresponds to the 
 top 2 inception blocks in the inceptionv3 arch
 """
 for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
  layer.trainable = False
 for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
  layer.trainable = True
 model.compile(
               optimizer=SGD(lr=0.0001, momentum=0.9), 
               loss='categorical_crossentropy', 
               metrics=['accuracy'])

  
def plot_training(history):
 acc = history.history['acc']
 val_acc = history.history['val_acc']
 loss = history.history['loss']
 val_loss = history.history['val_loss']
 epochs = range(len(acc))
 plt.plot(epochs, acc, 'r.')
 plt.plot(epochs, val_acc, 'r')
 plt.title('Training and validation accuracy')
 plt.figure()
 plt.plot(epochs, loss, 'r.')
 plt.plot(epochs, val_loss, 'r-')
 plt.title('Training and validation loss')
 plt.show()
  
train_dir="/home/cdsw/sample_set/train_dir"
val_dir="/home/cdsw/sample_set/val_dir"

nb_train_samples = get_nb_files(train_dir)
nb_classes = len(glob.glob(train_dir + "/*"))
nb_val_samples = get_nb_files(val_dir)
                           
nb_epoch = int(2)
batch_size = int(32)

train_datagen =  ImageDataGenerator(
                                    preprocessing_function=preprocess_input,
                                    rotation_range=30,width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator( preprocessing_function=preprocess_input,
                                  rotation_range=30,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
                                                    train_dir,
                                                    target_size=(IM_WIDTH, IM_HEIGHT),
                                                    batch_size=batch_size,)

validation_generator = test_datagen.flow_from_directory(
                                                        val_dir,
                                                        target_size=(IM_WIDTH, IM_HEIGHT),
                                                        batch_size=batch_size,)

base_model = InceptionV3(
                         weights='imagenet', 
                         include_top=False)

model = add_new_last_layer(
                           base_model, 
                           nb_classes)  

setup_to_transfer_learn(
                        model, 
                        base_model)

history_tl = model.fit_generator(
                                 train_generator,
                                 steps_per_epoch=nb_train_samples // 32,
                                 epochs=5,
                                 validation_data=validation_generator,
                                 validation_steps=nb_val_samples //32)

setup_to_finetune(model)

history_ft = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // 32,
    nb_epoch=5,
    validation_data=validation_generator,
    validation_steps=nb_val_samples //32,
    class_weight='auto')

model.save("/home/cdsw/model_3_learn")
plot_training(history_ft)

elapsed = (dt.datetime.now() - startTime)
print elapsed

"""Ref: http://bit.ly/2i4ZraH"""

                      
                      
                      
                      
                      
                      
                      
                      
                      
                      

                           
                           
