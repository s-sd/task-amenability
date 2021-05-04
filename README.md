# Task amenable data selection

Authors: 

Image quality assessment (IQA) is useful in medicine and beyond as it helps to ensure that a target task intended for an image can be performed reliably. General IQA methods may not reflect the usefulness of images for specific tasks and thus we propose to use a task-specific definition of image quality. We refer to this task-specific quality as "task amenability". 

In our work [Saeed et al. 2021](https://arxiv.org/abs/2102.07615) and [** Insert link to journal paper], we introduce the concept of task amenability and a reinforcement learning-based meta-learning framework to learn it. In this repository we provide the reinforcement leanring (RL) environment which allows for task amenability to be learnt. We also provide two interfaces, for two RL algorithms ([deep deterministic policy gradient (DDPG)](https://arxiv.org/abs/1509.02971) and [proximal policy optimisation (PPO)](https://arxiv.org/abs/1707.06347)), which allows for different datasets and target tasks to be used in the task amenability framework. Please see `example_ddpg.py` and `example_ppo.py` for examples of how custom datasets and target tasks can be used to train RL controllers to learn task amenability using the environment provided in `envs/task_amenability.py`. Note that we use classes defined in `interface.py` for these examples which provides an interface to interact with the environment. For instructions on how to use custom architectures for the DDPG or PPO algorithms, please refer to comments within `interface.py`.
