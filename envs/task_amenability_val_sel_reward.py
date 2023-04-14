import gym
import numpy as np
from gym import spaces

from keras import layers
import keras



class TaskAmenability(gym.Env):
    
    def __init__(self, x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape, val_sel_ratio=0.9):
        
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_holdout, self.y_holdout = x_holdout, y_holdout
                
        self.img_shape = img_shape
        
        
        self.task_predictor = task_predictor
        
        self.val_sel_ratio = val_sel_ratio
        
        
        self.controller_batch_size = 64
        self.task_predictor_batch_size = 32
        self.epochs_per_batch = 2
        
        self.img_shape = img_shape
        
        self.num_val = len(self.x_val)
        
        self.observation_space =  spaces.Box(low=0, high=1, shape=self.img_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.actions_list = []
        self.val_metric_list = [0.5]*10
        
        self.sample_num_count = 0
        
    def get_batch(self):
        shuffle_inds = np.random.permutation(len(self.y_train))
        self.x_train, self.y_train = self.x_train[shuffle_inds], self.y_train[shuffle_inds]
        return self.x_train[:self.controller_batch_size], self.y_train[:self.controller_batch_size]

    def compute_moving_avg(self):
        self.val_metric_list = self.val_metric_list[-10:]
        moving_avg = np.mean(self.val_metric_list)
        return moving_avg
    
    def select_samples(self, actions_list):
        actions_list = np.clip(actions_list, 0, 1)
        selection_vector = np.random.binomial(1, actions_list)
        logical_inds = [bool(elem) for elem in selection_vector]
        return self.x_train_batch[logical_inds], self.y_train_batch[logical_inds]
    
    def get_val_acc_vec(self):
        val_acc_vec = []
        for i in range(len(self.y_val)):
            metrics = self.task_predictor.evaluate(self.x_val[i:i+1], self.y_val[i:i+1], verbose=0)
            val_metric = metrics[-1]
            val_acc_vec.append(val_metric)
        return np.array(val_acc_vec)
    
    def get_val_sel_reward(self):
        sorted_inds = np.argsort(np.squeeze(self.val_sel_vec_normalised))
        num_rejected = int((1 - self.val_sel_ratio) * len(self.val_acc_vec))
        selected_val_accs = self.val_acc_vec[sorted_inds[num_rejected:]]
        return np.mean(selected_val_accs)
    
    def step(self, action):
        self.actions_list.append(action)
        self.sample_num_count += 1
        
        # print(self.sample_num_count)
        
        if self.sample_num_count < self.controller_batch_size+self.num_val:
            reward = 0
            done = False
            return self.x_data[self.sample_num_count], reward, done, {}
        
        else:
            x_train_selected, y_train_selected = self.select_samples(self.actions_list[:self.controller_batch_size])
            if len(y_train_selected) < 1:
                reward = -1
                done = True
            else:
                moving_avg = self.compute_moving_avg()
                
                self.task_predictor.fit(x_train_selected, y_train_selected, batch_size=self.task_predictor_batch_size, epochs=self.epochs_per_batch, shuffle=True, verbose=0)
                
                val_acc_vec = np.array(self.get_val_acc_vec())
                val_sel_vec = self.actions_list[self.controller_batch_size:]
                val_sel_vec_normalised = np.array(val_sel_vec) / np.mean(val_sel_vec)
                
                self.val_sel_vec_normalised = val_sel_vec_normalised
                self.val_acc_vec = val_acc_vec
                                
                val_metric = self.get_val_sel_reward() 
                                
                self.val_metric_list.append(val_metric)
                reward = val_metric - moving_avg
                done = True
            return np.random.rand(self.img_shape[0], self.img_shape[1], self.img_shape[2]), reward, done, {}
        
    def reset(self):
        self.x_train_batch, self.y_train_batch = self.get_batch()
        
        self.x_data = np.concatenate((self.x_train_batch, self.x_val), axis=0)
        self.y_data = np.concatenate((self.y_train_batch, self.y_val), axis=0)
        
        self.actions_list = []
        self.sample_num_count = 0

        return self.x_train_batch[self.sample_num_count]
    
    def save_task_predictor(self, task_predictor_save_path):
        self.task_predictor.save(task_predictor_save_path)
