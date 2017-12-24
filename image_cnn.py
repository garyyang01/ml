import keras
from keras.models import load_model
from PIL import Image
from keras.datasets import mnist
import numpy as np

model = load_model('mnist_cnn.h5')

im = Image.open('./9.bmp')
im2arr = np.array(im)
print (im2arr)
im2arr=im2arr/255
im2arr=im2arr.reshape(1,28,28,1) 
ans=model.predict(im2arr)[0]



bestclass = ''
bestconf = -1

for n in [0,1,2,3,4,5,6,7,8,9]:
	if (ans[n] > bestconf):
		bestclass = str(n)
		bestconf = ans[n]

print (ans)
print ("class : "+str(bestclass)+" with "+str(bestconf)+" confidence.")

