# Task amenable data selection

Authors: 

Image quality assessment (IQA) is useful in medicine and beyond as it helps to ensure that a target task intended for an image can be performed reliably. General IQA methods may not reflect the usefulness of images for specific tasks and thus we propose to use a task-specific definition of image quality. We refer to this task-specific quality as "task amenability". 

In our work [Saeed et al. 2021](https://arxiv.org/abs/2102.07615) and [** Insert link to journal paper], we introduce the concept of task amenability and a reinforcement learning-based meta-learning framework to learn it. In this repository we provide the reinforcement leanring (RL) environment which allows for task amenability to be learnt. We also provide two interfaces, for two RL algorithms ([deep deterministic policy gradient (DDPG)](https://arxiv.org/abs/1509.02971) and [proximal policy optimisation (PPO)](https://arxiv.org/abs/1707.06347)), which allows for different datasets and target tasks to be used in the task amenability framework. Please see `example_ddpg.py` and `example_ppo.py` for examples of how custom datasets and target tasks can be used to train RL controllers to learn task amenability using the environment provided in `envs/task_amenability.py`. Note that we use classes defined in `interface.py` for these examples which provides an interface to interact with the environment. Custom architectures for the DDPG or PPO algorithms can be used by modifying `interface.py`. The examples can be run from the root directory of this repository.

This repository also contains the code for an experiment in [** Insert link to journal paper], in which the PneumoniaMNIST dataset is used. We corrupt this dataset artificially and use the proposed task amenability framework to learn task amenability for a target task of pneumonia diagnosis. The code for the experiment is contained in `pneumoniamnist_experiment.py` and `pneumoniamnist_experiment_test_final_models.py` files. The final trained model and results along with the artificially corrupted data are contained in the directory `pneumoniamnist_experiment`. Samples of the images are presented below where a 'low controller score' means a controller predicted value below the 20th percentile and 'high controller score' means above. The second figure shows performance at different rejection ratios (ratio of samples with the lowest controller predicted values removed).

![alt text](pneumoniamnist_experiment/samples.png)

![alt text](pneumoniamnist_experiment/plot.png)
