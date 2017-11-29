import numpy as np
import cv2

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.datasets import cifar10
from keras.optimizers import Adam
import keras

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

STAMP = 'face_detection'
batch_size = 32
epochs = 100


def load_celeba(num_batches):
	"""
	Loads the celebA images and bounding boxes and splits them into training 
	and validation set.

	Arguments:
		num_batches: The number of batches to load (each batch includes 1000-2000 images)
	Returns:
		X_train: The training images
		Y_train: The trainign bounding boxes
		X_valid: The validation images
		Y_valid: The validation bounding boxes
	"""

	X_data = []
	bboxes = []
	for i in range(num_batches):
		X_data_batch = np.load('data/images/celebA_images'+str(i)+'.npy')
		bboxes_batch = np.load('data/bboxes/celebA_bboxes'+str(i)+'.npy')
		if bboxes_batch.shape[0] == 0:
			continue
		X_data.append(X_data_batch)
		bboxes.append(bboxes_batch)

	X_data = np.concatenate(X_data,axis=0)
	bboxes = np.concatenate(bboxes,axis=0)

	X_train, X_valid, Y_train, Y_valid = train_test_split(X_data, bboxes,
		test_size=0.2,
		random_state=42)

	return X_train, Y_train, X_valid, Y_valid


def build_model(img_width,
	img_height,
	channels,
	lr=1e-5,
	freeze=False):
	"""
	Loads the pretrained vgg16 model and weights and adds the output layers for face detection.
	VGG layers can either be frozen or trainable.

	Arguments:
		img_width:
		img_height:
		channels: The number of input channels.
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
	predictions = Dense(4, activation='relu')(drop1)

	model = Model(inputs=vgg_model.input, outputs=predictions)
	
	if freeze:
		for layer in vgg_model.layers:
			layer.trainable = False
	
	model.summary()
	adam = Adam(lr=lr, decay=1e-6)
	model.compile(optimizer=adam, loss='mse')

	return model


x_train, y_train, x_test, y_test = load_celeba(num_batches=3)
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
	lr=1e-5)

model_json = model.to_json()
with open('model/' + STAMP + ".json", "w") as json_file:
    json_file.write(model_json)

early_stopping =EarlyStopping(monitor='val_loss',
    patience=10)
bst_model_path = 'model/' + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True)

model.fit(x_train, y_train,
	batch_size=batch_size,
	epochs=epochs,
	validation_data=(x_test, y_test),
	shuffle=True,
	callbacks=[early_stopping, model_checkpoint])

'''
preds = model.predict(x_test)

for c,(img, pred, lab) in enumerate(zip(x_test_eval, preds, y_test)):
	print pred, lab
	x_1,y_1,width,height = pred.astype(int)
	predicted_roi = img[y_1:y_1+height,x_1:x_1+width,:]
	cv2.imshow('image',img)
	cv2.imshow('roi',predicted_roi)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	break
'''