import numpy as np

from copy import deepcopy

def add_rand_noise(img, noise):
    img_copy = deepcopy(img)
    img_copy = np.squeeze(img_copy)
    num_curropted = int(noise*np.shape(img)[0]*np.shape(img)[1])
    rand_0 = np.random.randint(low=0, high=np.shape(img)[0], size=(num_curropted))
    rand_1 = np.random.randint(low=0, high=np.shape(img)[1], size=(num_curropted))
    rand_noise = np.random.rand(num_curropted)
    img_copy[rand_0, rand_1] = rand_noise[:]
    return np.expand_dims(img_copy, axis=-1)

def add_rand_occlusion(img, occlusion):
    img_copy = deepcopy(img)
    img_copy = np.squeeze(img_copy)
    pix_val = int(np.shape(img)[0]*occlusion)
    edge = np.random.randint(0, 4)
    if edge==0:
        img_copy[:pix_val, :] = 0
    elif edge==1:
        img_copy[:, :pix_val] = 0
    elif edge==2:
        img_copy[-pix_val:, :] = 0
    elif edge==3:
        img_copy[:, -pix_val:] = 0
    return np.expand_dims(img_copy, axis=-1)

def corrupt_img(img):
    operation = np.random.choice([0, 1], p=[0.7, 0.3]) # 0 is rand noise, 1 is rand occlusion
    if int(operation)==0:
        noise = np.random.random() * 0.6
        img = add_rand_noise(img, noise)
    elif int(operation)==1:
        occlusion = np.random.random() * 0.6
        img = add_rand_occlusion(img, occlusion)
    return img

def corrupt_batch(imgs):
    corrupted_imgs = np.zeros(np.shape(imgs))
    for i in range(len(imgs)):
        corrupted_imgs[i] = corrupt_img(imgs[i])
    return corrupted_imgs

data_save_path = r'/home/s-sd/Desktop/task_amenability_repo/data/original/pneumoniamnist.npz'
numpy_zip = np.load(data_save_path)
x_train, y_train = numpy_zip['train_images'], numpy_zip['train_labels']
x_val, y_val = numpy_zip['val_images'], numpy_zip['val_labels']
x_holdout, y_holdout = numpy_zip['test_images'], numpy_zip['test_labels']

x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)
x_holdout = np.expand_dims(x_holdout, axis=-1)

x_train = x_train*(1/255.0)
x_val = x_val*(1/255.0)
x_holdout = x_holdout*(1/255.0)

x_train_corrupted = corrupt_batch(x_train)
x_val_corrupted = corrupt_batch(x_val)
x_holdout_corrupted = corrupt_batch(x_holdout)


import matplotlib.pyplot as plt
n=5
plt.imshow(x_train_corrupted[n]) ; n+=1


np.savez_compressed(r'/home/s-sd/Desktop/task_amenability_repo/data/corrupted/pneumoniamnist_corrupted.npz',
                    x_train=x_train_corrupted,
                    y_train=y_train,
                    x_val=x_val_corrupted,
                    y_val=y_val,
                    x_holdout=x_holdout_corrupted,
                    y_holdout=y_holdout)

numpy_zip2 = np.load(r'/home/s-sd/Desktop/task_amenability_repo/data/corrupted/pneumoniamnist_corrupted.npz')

x_train, y_train = numpy_zip2['x_train'], numpy_zip2['y_train']
x_val, y_val = numpy_zip2['x_val'], numpy_zip2['y_val']
x_holdout, y_holdout = numpy_zip2['x_holdout'], numpy_zip2['y_holdout']

import matplotlib.pyplot as plt
n=5
plt.imshow(x_holdout[n]) ; n+=1


