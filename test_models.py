import cv2
import numpy as np

from keras.models import model_from_json
from keras.datasets import cifar10

from vgg_cifar10 import load_cifar10_data
from vgg_face_detection import load_celeba


model_selection = raw_input('Choose simple/vgg_class/vgg_regr: ')

if model_selection == 'simple':
	STAMP = 'simple_cnn'
	(x_train, y_train), (x_val, y_val) = cifar10.load_data()
	print(x_val.shape[0], 'test samples')

elif model_selection == 'vgg_class':
	STAMP = 'vgg_cifar10'
	x_train, y_train, x_val, y_val = load_cifar10_data()

elif model_selection == 'vgg_regr':
	STAMP = 'face_detection'
	x_train, y_train, x_val, y_val = load_celeba(num_batches=1)

else:
	print('Model should be one of simple/vgg_class/vgg_regr')


x_val_orig = x_val
x_val = x_val.astype('float32')
x_val /= 255

json_file = open('model/' + STAMP + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('model/' + STAMP + '.h5')
print("Loaded model from disk")

preds = model.predict(x_val)

for c,(img, pred, lab) in enumerate(zip(x_val_orig, preds, y_val)):
	if model_selection == 'simple':
		pred = np.argmax(pred)
		print('Predicted class: %d' % pred)
		print('Original class: %d' % lab)
		cv2.imshow('image',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		break

	elif model_selection == 'vgg_class':
		pred = np.argmax(pred)
		cv2.imshow('image',img)
		print('Predicted class: %d' % pred)
		print('Original class: %d' % lab)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		break

	elif model_selection == 'vgg_regr':
		x_1,y_1,width,height = pred.astype(int)
		predicted_roi = img[y_1:y_1+height,x_1:x_1+width,:]
		cv2.imshow('image',img)
		cv2.imshow('roi',predicted_roi)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		break