# Sample-Efficient Decision Transformer (SE-DT)


## Overview

This paper generally builds upon [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345). We use the official codes as a guideline and also take several other github repositories as [reference](https://github.com/nikhilbarhate99/min-decision-transformer) to construct a minimized decision transformer implementation. Our codes are also published here on open-source platform [github](https://https://github.com/BillChan226/Sample-Efficient-Decision-Transformer).

## Results

In this study, we mainly conduct experiments to compare the performances of the original decision transformers(denoted as "DT") and our modified version of Sample-efficient decision transformers(denoted as "SE-DT") on the same tasks. The accumulated rewards along training are depicted above for each individual task. Table 1 summarizes the final performances of the sample-efficient decision transformer model(in terms of mean and variance). The convergence speed of accumulated rewards that sample-efficient decision transformers obtain for each environment are generally higher than that of the original decision transformer. It can be clear denoted that "SE-DT" demonstrates a better sample efficiency than "DT". However the variance of "SE-DT" is somehow larger. This is probably because the three terms of loss function are slightly different in their optimizing objectives, leading to a competitive variance. It is noted that the decision transformer model also demonstrates excellent performances on the untested Ant environment.

The training results on the high-dimensional humanoid robot InMoov is presented in Figure 11. While the action space of InMoov environment is high dimensional, the task is as simple as solving a linear optimization problem, which has a fixed solution. Therefore both models converge to the same accuracy. However, a slight triumph of "SE-DT" over "DT" in sample efficiency can still be observed, as the red curve is slightly above the upper edge of the blue curve. The results in InMoov environment validate that decision transformers could adapt to complex high dimensional robotic scenarios.

## Originality Claim

The implementation of the sample-efficient decision transformer is built upon the general pipeline of [min-decision-transformer](https://github.com/nikhilbarhate99/min-decision-transformer). I have modified the model.py file in decision_tranformer folder, so that the modified model can predict the states and returns-to-go at each timestep with respect to the input previous walk data. This is as opposed to the original transformer, whose forward function can only predict actions and returns-to-go. Additionally, I have modified the main entrance file train.py in scripts folder, so that in the main loop the model could calculate and back-propagate the error not only on action predictions but also on states and returns-to-go predictions. In this way, we expect the modified model to be more sample-efficient. 



For datasets, we simply test and compare our modified sample-efficient decision transformer to the original decision transformer on 4 benchmark dynamic mujoco environments and a high-dimensional humanoid robot InMooV that I designed with my previous lab partners. The mujoco environments can be easily set up through openai gym, and the InMooV environment settings is clarified in [this paper](https://iopscience.iop.org/article/10.1088/1742-6596/1746/1/012035/pdf). All the environment and dataset are publicly accessible.



## Run the experiments

### Mujoco-py

Install `mujoco-py` library by following instructions on [mujoco-py repo](https://github.com/openai/mujoco-py)


### D4RL Data

Datasets are expected to be stored in the `data` directory. Install the [D4RL repo](https://github.com/rail-berkeley/d4rl). Then save formatted data in the `data` directory by running the following script:
```
python3 data/download_d4rl_datasets.py
```


### Running experiments

- Example command for training:
```
python3 scripts/train.py --env halfcheetah --dataset medium --device cuda
```


- Example command for testing with a pretrained model:
```
python3 scripts/test.py --env halfcheetah --dataset medium --device cpu --num_eval_ep 1 --chk_pt_name dt_halfcheetah-medium-v2_model_22-02-13-09-03-10_best.pt
```
The `dataset` needs to be specified for testing, to load the same state normalization statistics (mean and var) that is used for training.
An additional `--render` flag can be passed to the script for rendering the test episode.


- Example command for plotting graphs using logged data from the csv files:
```
python3 scripts/plot.py --env_d4rl_name halfcheetah-medium-v2 --smoothing_window 5
```
Additionally `--plot_avg` and `--save_fig` flags can be passed to the script to average all values in one plot and to save the figure.


### Note:
1. If you find it difficult to install `mujoco-py` and `d4rl` then you can refer to their installation in the colab notebook
2. Once the dataset is formatted and saved with `download_d4rl_datasets.py`, `d4rl` library is not required further for training.
3. The evaluation is done on `v3` control environments in `mujoco-py` so that the results are consistent with the decision transformer paper.

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

[2] Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M., Dudzik, A., Chung, J., ... & Silver, D. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. *Nature*, *575*(7782), 350-354.

[3] Parisotto, E., Song, F., Rae, J., Pascanu, R., Gulcehre, C., Jayakumar, S., ... & Hadsell, R. (2020, November). Stabilizing transformers for reinforcement learning. In International conference on machine learning (pp. 7487-7498). PMLR.

[4] Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., ... & Mordatch, I. (2021). Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34, 15084-15097.

[5] Janner, M., Li, Q., & Levine, S. (2021, June). Reinforcement learning as one big sequence modeling problem. In ICML 2021 Workshop on Unsupervised Reinforcement Learning.
