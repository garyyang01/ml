import keras
from keras.models import load_model
from PIL import Image
from keras.datasets import mnist
import numpy as np

model = load_model('mnist_cnn.h5')	#讀取model
im_address="image.bmp"				#圖片路徑
im = Image.open(im_address) 		
#資料前處理
im2arr = np.array(im)
im2arr=im2arr/255
im2arr=im2arr.reshape(1,28,28,1) 
ans=model.predict(im2arr)[0]

#one hot decoder
bestclass = ''
bestconf = -1
for n in [0,1,2,3,4,5,6,7,8,9]:
	if (ans[n] > bestconf):
		bestclass = str(n)
		bestconf = ans[n]
print (ans)#輸出結果信賴矩陣
print ("picture: "+im_address+" assigned to class "+str(bestclass)+" with "+str(bestconf)+" confidence.")#輸出結果

