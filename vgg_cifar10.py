import numpy as np
import cv2

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint


batch_size = 32
num_classes = 10
epochs = 100
STAMP = 'vgg_cifar10'


def load_cifar10_data(img_rows=48,
	img_cols=48):
	"""
	Loads the cifar10 data set and resizes the images to fit the minimum vgg input. Also converts
	the labels to categoricals.

	Arguments:
		img_rows: the new img height (vgg can take 48 to 299)
		img_cols: the new img width (vgg can take 48 to 299)
	Returns:
		X_train: The training images
		Y_train: The trainign labels
		X_valid: The validation images
		Y_valid: The validation labels
	"""

	(X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

	X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:,:,:,:]])
	X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:,:,:,:]])

	Y_train = to_categorical(Y_train[:], num_classes)
	Y_valid = to_categorical(Y_valid[:], num_classes)

	return X_train, Y_train, X_valid, Y_valid


def build_model(img_width,
	img_height,
	channels,
	num_classes,
	lr=1e-5,
	freeze=False):
	"""
	Loads the pretrained vgg16 model and weights and adds the output layers for classification.
	VGG layers can either be frozen or trainable.

	Arguments:
		img_width:
		img_height:
		channels: The number of input channels.
		num_classes: The number of classes to predict.
		lr: The learning rate.
		freeze: If True then the vgg conv layers are untrainable.
	Returns:
		model: The compiled keras model
	"""

	vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, channels))

	vgg_output = vgg_model.output
	drop0 = Dropout(0.5)(vgg_output)
	flat = Flatten()(drop0)
	dense1 = Dense(512, activation='relu')(flat)
	drop1 = Dropout(0.5)(dense1)
	predictions = Dense(num_classes, activation='softmax')(drop1)

	model = Model(inputs=vgg_model.input, outputs=predictions)
	
	if freeze:
		for layer in vgg_model.layers:
			layer.trainable = False

	model.summary()
	adam = Adam(lr=lr, decay=1e-6)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


x_train, y_train, x_test, y_test = load_cifar10_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

img_width, img_height, channels = x_train.shape[1],x_train.shape[2],x_train.shape[3]

model = build_model(img_width=img_width,
	img_height=img_height,
	channels=channels,
	num_classes=num_classes,
	lr=1e-5)

model_json = model.to_json()
with open('model/' + STAMP + ".json", "w") as json_file:
    json_file.write(model_json)

early_stopping =EarlyStopping(monitor='val_loss',
    patience=10)

bst_model_path = 'model/' + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    save_weights_only=True)

model.fit(x_train, y_train,
	batch_size=batch_size,
	epochs=epochs,
	validation_data=(x_test, y_test),
	shuffle=True,
	callbacks=[early_stopping, model_checkpoint])