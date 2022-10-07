import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K

class MetaRL1(gym.Env):
    def __init__(self):
        
        self.batch_size = 1000
        self.dsnet_batch_size = 128
        self.num_val = 1123
        
        self.img_shape = (96, 96, 1)
        
        save_path =  r'/home/s-sd/Desktop/qa_separate_labels/fin_data'
        
        holdout_imgs_loaded = np.load(os.path.join(save_path, 'test', 'imgs.npy'))
        holdout_labels_loaded = np.load(os.path.join(save_path, 'test', 'ss_seg.npy'))
        holdout_q_labels_loaded = np.load(os.path.join(save_path, 'test', 'q_seg.npy'))        
        val_imgs_loaded = np.load(os.path.join(save_path, 'val', 'imgs.npy'))
        val_labels_loaded = np.load(os.path.join(save_path, 'val', 'ss_seg.npy'))     
        train_imgs_loaded = np.load(os.path.join(save_path, 'train', 'imgs.npy'))
        train_labels_loaded = np.load(os.path.join(save_path, 'train', 'ss_seg.npy'))
        
        self.x_holdout, self.y_holdout, self.z_holdout = holdout_imgs_loaded, holdout_labels_loaded, holdout_q_labels_loaded
        self.x_train, self.y_train = train_imgs_loaded, train_labels_loaded
        
        self.x_val, self.y_val = val_imgs_loaded, val_labels_loaded
        
        
        self.x_val = np.array(self.x_val, dtype=np.float32)
        self.y_val = np.array(self.y_val, dtype=np.uint8)
        
        self.x_train = np.array(self.x_train, dtype=np.float32)
        self.y_train = np.array(self.y_train, dtype=np.uint8)
        
        self.x_holdout = np.array(self.x_holdout, dtype=np.float32)
        self.y_holdout = np.array(self.y_holdout, dtype=np.uint8)
        self.z_holdout = np.array(self.z_holdout, dtype=np.float32)
        
        
        self.x_val = np.expand_dims(self.x_val, axis=-1)
        self.y_val = np.expand_dims(self.y_val, axis=-1)
        
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.y_train = np.expand_dims(self.y_train, axis=-1)
        
        self.x_holdout = np.expand_dims(self.x_holdout, axis=-1)
        self.y_holdout = np.expand_dims(self.y_holdout, axis=-1)
        self.z_holdout = np.expand_dims(self.z_holdout, axis=-1)
        
        
        self.y_train = self.process_y_data(self.y_train)
        self.y_val = self.process_y_data(self.y_val)
        self.y_holdout = self.process_y_data(self.y_holdout)
        
        
        self.dsnet = self.build_dsnet(self.img_shape)
        self.dsnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.observation_space =  spaces.Box(low=0, high=1, shape=(258,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.val_metric_list = [0.0] * 10
        self.actions_list = []
        self.sample_num_count = 0
        
        self.prev_action = 0.0
        self.prev_reward = 0.0
        
        self.encoder = keras.models.load_model(r'/home/s-sd/Desktop/qa_meta_rl/encoder')
        
    def encode_img_and_prev(self, img, prev_action, prev_reward):
        encoding = self.encoder.predict(np.expand_dims(img, axis=0))
        encoding = np.array(np.squeeze(encoding))
                
        prev_action = np.reshape(prev_action, (1,))
        prev_reward = np.reshape(prev_reward, (1,))
        
        encoding_and_prev = np.append(encoding, prev_action, axis=0)
        encoding_and_prev = np.append(encoding_and_prev, prev_reward, axis=0)
        return encoding_and_prev
        
    def get_batch(self):
        shuffle_inds = np.random.permutation(len(self.y_train))
        self.x_train, self.y_train = self.x_train[shuffle_inds, :, :, :], self.y_train[shuffle_inds]
        return self.x_train[:self.batch_size, :, :, :], self.y_train[:self.batch_size]
        
    def shuffle_imgs_labels(self, imgs, labels, q_labels):
        shuffle_inds = np.random.permutation(len(labels))
        imgs_shuffled = imgs[shuffle_inds, :, :, :]
        labels_shuffled = labels[shuffle_inds]
        q_labels_shuffled = q_labels[shuffle_inds]
        return imgs_shuffled, labels_shuffled, q_labels_shuffled
        
    def compute_moving_avg(self):
        self.val_metric_list = self.val_metric_list[-10:]
        moving_avg = np.mean(self.val_metric_list)
        # terminal condition here and raise keyboard interrupt
        return moving_avg
        
    def select_samples(self, actions_list):
        # print(self.actions_list)
        actions_list = np.clip(actions_list, 0, 1)
        selection_vector = np.random.binomial(1, actions_list)
        logical_inds = [bool(elem) for elem in selection_vector]
        return self.x_train_batch[logical_inds], self.y_train_batch[logical_inds]
    
    def get_val_acc_vec(self):
        val_acc_vec = []
        for i in range(len(self.y_val)):
            metrics = self.dsnet.evaluate(self.x_val[i:i+1], self.y_val[i:i+1], verbose=0)
            val_metric = metrics[-1]
            val_acc_vec.append(val_metric)
        return np.array(val_acc_vec)
        
    def step(self, action):
        self.actions_list.append(action)
        self.sample_num_count += 1
        
        self.prev_action = 0.0 if len(self.actions_list)<2 else self.actions_list[-2]
        self.prev_action = np.clip(self.prev_action, a_min=0.0, a_max=1.0)
        
        if self.sample_num_count < self.batch_size+self.num_val:
            reward = 0
            done = False
            obs = self.encode_img_and_prev(self.x_data[self.sample_num_count, :, :, :], self.prev_action, self.prev_reward)
            # print(np.shape(obs))
            return obs, reward, done, {}
        else:
            x_train_selected, y_train_selected = self.select_samples(self.actions_list[:self.batch_size])
            if len(y_train_selected) < 1:
                reward = -1
                done = True
                obs = self.encode_img_and_prev(np.random.rand(96, 96, 1), self.prev_action, self.prev_reward)
            else:
                moving_avg = self.compute_moving_avg()
                self.dsnet.fit(x_train_selected, y_train_selected, batch_size=self.dsnet_batch_size, epochs=2, shuffle=True, verbose=0)
                
                val_acc_vec = self.get_val_acc_vec()
                val_sel_vec = self.actions_list[self.batch_size:]
                val_sel_vec_normalised = np.array(val_sel_vec) / np.mean(val_sel_vec)
                
                val_metric = np.mean(np.multiply(val_sel_vec_normalised, np.array(val_acc_vec)))
                
                self.val_metric_list.append(val_metric)
                reward = val_metric - moving_avg
                done = True
                print('\n', val_metric, moving_avg)
                obs = self.encode_img_and_prev(np.random.rand(96, 96, 1), self.prev_action, self.prev_reward)
            return obs, reward, done, {}
        

    def reset(self):
        self.x_train_batch, self.y_train_batch = self.get_batch()
        
        self.x_data = np.concatenate((self.x_train_batch, self.x_val), axis=0)
        self.y_data = np.concatenate((self.y_train_batch, self.y_val), axis=0)
        
        self.actions_list = []
        self.sample_num_count = 0
        
        self.prev_action = 0.0
        self.prev_reward = 0.0

        obs = self.encode_img_and_prev(self.x_train_batch[self.sample_num_count, :, :, :], self.prev_action, self.prev_reward)
        return obs

    def build_dsnet(self, input_shape):
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
        
    def process_y_data(self, y_data):
        y_data_processed = np.zeros(np.shape(y_data)[0])
        for i in range(np.shape(y_data)[0]):
            slice_max = np.amax(y_data[i])
            y_data_processed[i] = slice_max
        return y_data_processed
