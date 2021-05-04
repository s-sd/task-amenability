
from interface import DDPGInterface

from keras import layers
import keras

import numpy as np
import os

import matplotlib.pyplot as plt



def build_task_predictor(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

img_shape = (28, 28, 1)

numpy_zip = np.load(r'pneumoniamnist_experiment/pneumoniamnist_corrupted.npz')

x_train, y_train = numpy_zip['x_train'], numpy_zip['y_train']
x_val, y_val = numpy_zip['x_val'], numpy_zip['y_val']
x_holdout, y_holdout = numpy_zip['x_holdout'], numpy_zip['y_holdout']

num_train_samples = len(y_train)
num_val_samples = len(y_val)
num_holdout_samples = len(y_holdout)


task_predictor = build_task_predictor(img_shape)
task_predictor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # speciffy the loss and metric used to train target net and controller respectively


def build_actor_critic(img_shape, action_shape=(1,)):
    
    n_actions = action_shape[0]
    
    act_in = layers.Input((1,) + img_shape)
    act_in_reshape = layers.Reshape((img_shape))(act_in)
    act_x = layers.Conv2D(32, (3,3), activation='relu')(act_in_reshape)
    act_x = layers.MaxPool2D((2,2))(act_x)
    act_x = layers.Conv2D(64, (3,3), activation='relu')(act_x)
    act_x = layers.MaxPool2D((2,2))(act_x)
    act_x = layers.Conv2D(64, (3,3), activation='relu')(act_x)
    act_x = layers.Flatten()(act_x)
    act_x = layers.Dense(64, activation='relu')(act_x)
    act_x = layers.Dense(32, activation='relu')(act_x)
    act_x = layers.Dense(16, activation='relu')(act_x)
    act_out = layers.Dense(n_actions, activation='sigmoid')(act_x)
    actor = keras.Model(inputs=act_in, outputs=act_out)
    
    action_input = layers.Input(shape=(n_actions,), name='action_input')
    observation_input = layers.Input((1,) + img_shape, name='observation_input')
    observation_input_reshape = layers.Reshape((img_shape))(observation_input)
    observation_x = layers.Conv2D(32, (3,3), activation='relu')(observation_input_reshape)
    observation_x = layers.MaxPool2D((2,2))(observation_x)
    observation_x = layers.Conv2D(64, (3,3), activation='relu')(observation_x)
    observation_x = layers.MaxPool2D((2,2))(observation_x)
    observation_x = layers.Conv2D(64, (3,3), activation='relu')(observation_x)
    flattened_observation = layers.Flatten()(observation_x)
    x = layers.Concatenate()([action_input, flattened_observation])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(1)(x)
    critic = keras.Model(inputs=[action_input, observation_input], outputs=x)
    return actor, critic, action_input


controller_batch_size = 512
task_predictor_batch_size = 256

num_train_episodes = 512

save_path = r'pneumoniamnist_experiment/final_models/pneumoniamnist_experiment_train_session'

controller_weights_save_path = save_path + 'controller_episode_' + str(num_train_episodes)
task_predictor_save_path = save_path + 'task_predictor_episode_' + str(num_train_episodes)

actor, critic, action_input = build_actor_critic(img_shape)

interface = DDPGInterface(x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape, 
                          load_models=True, controller_weights_save_path=controller_weights_save_path, task_predictor_save_path=task_predictor_save_path,
                          custom_controller=True, actor=actor, critic=critic, action_input=action_input,
                          modify_env_params=True, modified_env_params_list=[controller_batch_size, task_predictor_batch_size])

holdout_controller_preds = interface.get_controller_preds_on_holdout()

def reject_lowest_controller_valued_samples(rejection_ratio, holdout_controller_preds, x_holdout, y_holdout):
    sorted_inds = np.argsort(holdout_controller_preds)
    num_rejected = int(rejection_ratio * len(sorted_inds))
    selected_x_holdout, selected_y_holdout = x_holdout[sorted_inds[num_rejected:], :, :, :], y_holdout[sorted_inds[num_rejected:]]
    return selected_x_holdout, selected_y_holdout

def compute_mean_performance(x, y, interface):
    mean_performance_metric = interface.task_predictor.evaluate(x, y)
    return mean_performance_metric[-1]


performances = []
for rejection_ratio in np.arange(0.0, 0.5, 0.1):
    selected_x_holdout, selected_y_holdout = reject_lowest_controller_valued_samples(rejection_ratio, holdout_controller_preds, x_holdout, y_holdout)
    performance = compute_mean_performance(selected_x_holdout, selected_y_holdout, interface)
    performances.append(performance)

plt.plot(performances)


