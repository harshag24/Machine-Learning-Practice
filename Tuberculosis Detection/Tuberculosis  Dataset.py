import tensorflow as tf
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
from PIL import Image
import cv2
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt


#Used to check average dimension of images for resizing
# path_0 = 'C:/Users/harsh/OneDrive/Desktop/Tuberculosis Datasets/Final Dataset/0/'
# path_1 = 'C:/Users/harsh/OneDrive/Desktop/Tuberculosis Datasets/Final Dataset/1/'

# total_h = 0
# total_w=0
#
# for file in os.listdir(path_0):
#     im = cv2.imread(path_0 + file)
#     h, w, c = im.shape
#     total_h+= h
#     total_w+= w
#
# for file in os.listdir(path_1):
#     im = cv2.imread(path_1 + file)
#     h, w, c = im.shape
#     total_h+= h
#     total_w+= w

# print(total_h//800 , total_w//800) #Output - 3111 2971



#It was manually shifted in another folder for splitting between train and val so it won't work
#path_resized_0 = 'C:/Users/harsh/OneDrive/Desktop/Tuberculosis Datasets/Resized Dataset/0/'
#path_resized_1 = 'C:/Users/harsh/OneDrive/Desktop/Tuberculosis Datasets/Resized Dataset/1/'
# for file in os.listdir(path_0):
#     im = cv2.imread(path_0 + file)
#     im = cv2.resize(im , (3000,3000))
#     cv2.imwrite( path_resized_0 + file , im)
#
# for file in os.listdir(path_1):
#     im = cv2.imread(path_1 + file)
#     im = cv2.resize(im, (3000, 3000))
#     cv2.imwrite(path_resized_1 + file, im)

train_path = 'C:/Users/harsh/OneDrive/Desktop/Tuberculosis Datasets/Resized Dataset/Train/'
val_path = 'C:/Users/harsh/OneDrive/Desktop/Tuberculosis Datasets/Resized Dataset/Validation/'


train_data_generator = ImageDataGenerator(
		rescale=1/255,
		rotation_range=30,
		zoom_range=0.2,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		horizontal_flip=True,
		fill_mode="nearest")

val_data_generator = ImageDataGenerator(rescale=1/255)

train_gen=train_data_generator.flow_from_directory(train_path,
                                                   target_size=(3000,3000),
                                                   class_mode='binary',
                                                   batch_size=16,
                                                   shuffle=True)

validation_gen=train_data_generator.flow_from_directory(val_path,
                                                   target_size=(3000,3000),
                                                   class_mode='binary',
                                                   batch_size=16,
                                                   shuffle=True)


base_model=tf.keras.applications.InceptionV3(include_top=False,weights='imagenet',input_shape=(3000,3000,3))
base_model.trainable=False
model=tf.keras.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy',tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')])

class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if(logs['accuracy']>0.98):
      self.model.stop_training=True
callbacks=MyCallback()

history=model.fit(train_gen,validation_data=validation_gen,epochs=200,callbacks=callbacks,verbose=1)




#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
precision=history.history['precision']
val_precision=history.history['val_precision']
recall=history.history['recall']
val_recall=history.history['val_recall']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation Precision per epoch
#------------------------------------------------
plt.plot(epochs, precision, 'r', "Training Precision")
plt.plot(epochs, val_precision, 'b', "Validation Precision")
plt.title('Training and validation Precision')
plt.figure()

plt.plot(epochs, recall, 'r', "Training Recall")
plt.plot(epochs, val_recall, 'b', "Validation Recall")
plt.title('Training and validation Recall')