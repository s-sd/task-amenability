from stable_baselines import PPO2
import os
import numpy as np
import gym
from stable_baselines.common.vec_env import DummyVecEnv


from gym_metarl.envs.metarl0 import MetaRL0
from gym_metarl.envs.metarl1 import MetaRL1
from gym_metarl.envs.metarl2 import MetaRL2
from gym_metarl.envs.metarl3 import MetaRL3

os.environ['CUDA_VISIBLE_DEVICES']='0'

# this env just created to get parameters like batchsize, can be deleted later

def make_env():
    return MetaRL0()

# this env is just created to get params (the rest must be wrapped by dummyvecenv)
env = MetaRL0()
batch_size = env.batch_size + env.num_val

del env

env_1 = DummyVecEnv([MetaRL0])
env_2 = DummyVecEnv([MetaRL1])
env_3 = DummyVecEnv([MetaRL2])

# env_4 = DummyVecEnv([MetaRL3])

model = PPO2('MlpLstmPolicy', 
             env_1,
             nminibatches=1,
             n_steps=batch_size,
             gamma=1.0,
             verbose=2)

num_runs_per_env = 8


train_envs_list = [env_1, env_2, env_3]

# test_envs_list = [env_4]


num_trials = 160

env_1_metrs = []
env_2_metrs = []
env_3_metrs = []

save_freq = 20


env = train_envs_list[0]
dsnet = env.get_attr('dsnet')[0]



def rep_update(dsnet, old_weights, new_weights, eps):
    weights_to_set = dsnet.get_weights()
    for layer in range(len(new_weights)):
        weights_to_set[layer] = old_weights[layer]+((new_weights[layer]-old_weights[layer])*eps)
    dsnet.set_weights(weights_to_set)
    return dsnet



for i in range(num_trials):
        
    env_index = np.random.randint(low=0, high=3)
    env = train_envs_list[env_index]

    print(f'Trial {i+1}/{num_trials}: Env {env_index}')
    
    # reset env and set state to None which when passed to predict resets LSTM state
    obs = env.reset()
    state=None
            
    model.set_env(env)
    _, _ = model.predict(obs, state=state, mask=[False]) 
   
    env.set_attr('dsnet', dsnet)
   
    for i in range(num_runs_per_env):
        dsnet_copy = env.get_attr('dsnet')[0]
        dsnet_weights_old = env.get_attr('dsnet')[0].get_weights()
        model.learn(batch_size*1)
        dsnet_weights_new = env.get_attr('dsnet')[0].get_weights()
        frac_done = i/num_trials
        dsnet_copy_to_set = rep_update(dsnet_copy, dsnet_weights_old, dsnet_weights_new, eps=(1-frac_done)*1)
        env.set_attr('dsnet', dsnet_copy_to_set)
    
    
    dsnet = env.get_attr('dsnet')[0]
    
    env_1_metr = train_envs_list[0].get_attr('val_metric_list')[0][-1]
    env_2_metr = train_envs_list[1].get_attr('val_metric_list')[0][-1]
    env_3_metr = train_envs_list[2].get_attr('val_metric_list')[0][-1]
    
    env_1_metrs.append(env_1_metr)
    env_2_metrs.append(env_2_metr)
    env_3_metrs.append(env_3_metr)
    
    # print(f'\nTrial {i+1}:\n env_1:{env_1_metr}\n env_2:{env_2_metr}\n env_3:{env_3_metr}\n')
    
    # with open(r'/home/zcemsus/qa_project_2/qa_meta_rl_class_shrep/results.txt', 'a') as f:
    #     print(f'\nTrial {i+1}:\n env_1:{env_1_metr}\n env_2:{env_2_metr}\n env_3:{env_3_metr}\n', file=f)
    
    # if i%save_freq==0:
    #     model.save(f'/home/zcemsus/qa_project_2/qa_meta_rl_class_shrep/models/trial_{i}')
        
    #     dsnet.save(f'/home/zcemsus/qa_project_2/qa_meta_rl_class_shrep/dsnets/trial_{i}')
        
# first run will take around 15 minutes and then subsequent runs will be faster (approx 1 minute per run)



