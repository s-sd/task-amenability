from interface import PPOInterface

from keras import layers
import keras

import numpy as np

import os

def build_task_predictor(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

img_shape = (96, 96, 1)

num_train_samples = 100
num_val_samples = 50
num_holdout_samples = 50

x_train = np.random.rand(num_train_samples, img_shape[0], img_shape[1], img_shape[2])
y_train = np.random.randint(low=0, high=2, size=(num_train_samples, 1))

x_val = np.random.rand(num_val_samples, img_shape[0], img_shape[1], img_shape[2])
y_val = np.random.randint(low=0, high=2, size=(num_val_samples, 1))

x_holdout = np.random.rand(num_holdout_samples, img_shape[0], img_shape[1], img_shape[2])
y_holdout = np.random.randint(low=0, high=2, size=(num_holdout_samples, 1))


task_predictor = build_task_predictor(img_shape)
task_predictor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # speciffy the loss and metric used to train target net and controller respectively

interface = PPOInterface(x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape)

interface.train(6)

save_dir = 'temp'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

controller_save_path = r'temp/train_session_2_ppo_controller'
task_predictor_save_path = r'temp/train_session_2_task_predictor'
interface.save(controller_save_path=controller_save_path,
               task_predictor_save_path=task_predictor_save_path)


interface = PPOInterface(x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape, load_models=True, controller_save_path=controller_save_path, task_predictor_save_path=task_predictor_save_path)

interface.train(6)
