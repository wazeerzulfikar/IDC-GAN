from scipy import misc
import os

img_path = "./dataset"

save_path_rain = "./rain_images"
save_path_derain = "./derain_images"

if not os.path.exists(save_path_rain):
	os.mkdir(save_path_rain)
if not os.path.exists(save_path_derain):
	os.mkdir(save_path_derain)

count = 0

for filename in os.listdir(img_path):
	if filename.endswith('jpg'):
		img = misc.imread(os.path.join(img_path,filename))
		width = img.shape[1]
		img1 = img[:,:int(width/2),:]
		img2 = img[:,int(width/2):,:]
		misc.imsave(os.path.join(save_path_derain,"derain_%d.jpg"%count),img1)
		misc.imsave(os.path.join(save_path_rain,"rain_%d.jpg"%count),img2)
		count+=1
	if count%100==0:
		print("%d finished!"%count)
