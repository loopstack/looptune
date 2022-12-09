# LoopToolEnv project

To optimize ML workloads for custom architecture, generally there are tree options: 
* Hire an expert
* Develop iterative algorithmic approach
* Develop machine learning based approach

## Hire an expert
Hiring expert to tune knobs of compiler for specific architecture requires many engineer-hours, usually cannot be used for different architecture, and with increasing of complexity of tuning knobs most likely an expert will not come up with un optimal solution. Can we do better? Yes we can!

## Iterative algorithmic approach
For long time engineers figured out that they can use iterative improvement algorithms like hill climbing, genetic algorithms and others to boost performance of their application. Depending on optimization space and starting point, these techniques are often able to find better optimizations by just shoting hundreds of random actions and comparing their final performance. With the increasing of action space or optimization sequence length, this approach can explore only thiny piece of the space. To mitigate this problem we can use some kind of greedy algorithm that is chosing localy optimal action and keeping performence increasing each step. Unfortunatelly space of optimization is not convex and sometimes choosing actions that don't increase performance immediately can be beneficial in long turn. 

Ideally, with unlimited resources we could solve this problem by using lookahead of N steps to see if an action is going to be beneficial, and than pick the best action. In reality we don't really know how much of a lookahead will be enough and we would need to expand our search tree every step exponentially in N, before we make a decision. This could be very costly, and like in a case of expert, we cannot use learned optimization sequence to other problem.

Genetic algorithms are trying to explore the space of proising optimization by combining other promising sequences. This makes sense since some of optimizations go well together, but often times this is program dependent and it takes a lot of time to find beneficial longer optimization sequences. As previous approaches this fails to generalize on new problems. Can we do better. Absolutelly!

## Machine learning based approach
With the raise of deep machine learning it become possible to train networks to store knowledge in huge latent spaces, and predict output functions given input data. One of the first approaches that uses neural networks for code optimization is predicting cost of given code. With a right cost model, we would be able to use all search and genetic algorithms way faster since se don't need to evaluate programs but rather apply just forward propagation which is way cheper. As you could imagine, cost models are still not enough to completely solve our problem, since we are still need to run search which cannot explore full optimization space. 

Second approach is to model our problem in reinforcement learning setup and train network to learn the policy. Policy tells us what actions are the most beneficial given future reward. This is huge jump compared to cost model, since we don't need to think about lookahead and search algorithms. Following the right policy would always always lead us to the optimal solution. Even more, the right policy can with deep network can generalize accross different programs as well. 

Wohoooo, we have general approach to optimize any program! Let see if we can find the "right policy" :)


___
# LoopTool Environment

To map our problem to reinforcement learning lets see example of the problem we are trying to solve. 

<img src="docs/imgs/loop_tree.png" width="500">

In the figure above agent(cursor) starts on line 0 of loop nest and all actions are done on current loop. The agent moves from L0 to L6 with actions "up" and "down". Beside this agent can use "swap_up"/"swap_down" to swap current with the upper/lower loop. Agent can split the current loop into two loops with split power of 2 factor (from 2 to 1024). Agent can merge a loop so it fuses with the parent loops with the same iterator. Agent can annotate current loop with "unroll" and "vectorize" that low level compiler will use to optimize it.

For convenience action space is given here:
* up 
* down
* swap_up
* swap_down
* split_4
* split_8
* split_16


We defined several observation spaces including:
* runtime - execution time in seconds
* flops - FLOPs collected from execution
* ir - string that encodes loop nest
* loop_tree - human readable string of loop tree
* ir_tree_networkx - loop tree encoded in networkx as tree
* ir_graph_networkx - loop tree encoded in networkx as graph
* loops_tensor - torch tensor of features associated to each loop (padded to lt.LoopTreeAgent.max_loops)
* stride_tensor - 1d tensor (len 32) of frequency of memory accesses for stride (power of 2 strides)
* 5_prev_actions_tensor - 1d tensor of 1 hot encoding of 5 previous actions


___
## How to run

### 0. Setup environment variables:
```
export LOOP_TOOL_ROOT=$path_to_this_dir
export WANDB_CONSOLE=off
export MAX_GFLOPS=$peak_gflops_measurement
export RLLIB_NUM_GPUS=$num_gpus_available
```

### 1. Generate dateset:
```
python loop_tool_service/benchmarks/generator.py --kind=mm --dimA=64:128:16,64:128:16 --dimB=64:128:16,64:128:16  --out=/private/home/dejang/tools/loop_tool_env/loop_tool_service/benchmarks/mm64_128_16_range
# Register dataset
python setup.py install
```

### 2. Enable Wandb logging
Create Weight and Biases account and put your wandb key in $LOOP_TOOL_ROOT/wandb_key.txt


### 3. Train policy model:


On SLURM:

```
python loop_tool_service/models/rllib/launcher/slurm_launch.py --app=rllib_agent.py --time=1:00:00 -nc=80 -ng=2 --iter=1  --dataset=mm64_128_16_range  --trainer=dqn.ApexTrainer --steps=10  --eval_size=5 --eval_time=10

```

On local node:

```
python loop_tool_service/models/rllib/rllib_agent.py --iter=1 --dataset=mm64_256_16_range  --trainer=dqn.ApexTrainer  --eval_size=2 --eval_time=4
```

At the end of the training we print path to the evaluation of the LoopTune policy network.
```
...
Saved at:  $path-to-evaluation-directory
Return from train!
```

To see more login to Wandb and check out training curves and evaluation.

Additionally, you can start training from finished training logged on wandb with:

On SLURM:
```
python loop_tool_service/models/rllib/launcher/slurm_launch.py --app=rllib_agent.py --time=1:00:00 -nc=80 -ng=2 --iter=1  --dataset=mm64_128_16_range --trainer=dqn.ApexTrainer --steps=10  --eval_size=5 --eval_time=10 --wandb_url=$wandb_run_path
```
or locally:
```
python loop_tool_service/models/rllib/rllib_agent.py --iter=1 --dataset=mm64_256_16_range  --trainer=dqn.ApexTrainer  --eval_size=2 --eval_time=4 --wandb_url=$wandb_run_path

```


### 4. Evaluation:
Once the training is done you can reproduce evaluation with:
```
python loop_tool_service/experiments/compare_searches/search.py --trainer=dqn.ApexTrainer --wandb_url=$wandb_run_path
```
If you copy graph to online graphviz you can see visualization of 
searches.


## Comparison to Numpy and TVM

```
cd loop_tool_service/experiments/compare_tvm
python compare_tvm.py --size=25 --wandb_url=$wandb_run_path
```

# Datasets
Datasets are important part of training RL agent. It is important that they are organized from easy to hard. To generate dataset go to loop_tool_service/benchmarks directory and run generator.py. This will create an example or all permutations of matrix multiplication or convolution. With reader.py you can visualise each of these benchmarks.

```
python loop_tool_service/benchmarks/generator.py --kind=mm --dimA=128,128 --dimB=128,128 --out=$LOOP_TOOL_ROOT/loop_tool_service/benchmarks/mm128_16_8_128 --permute

# To register dataset run
python setup.py install
```

To examine any created benchmark just run:
```
python loop_tool_service/benchmarks/reader.py --bench=$LOOP_TOOL_ROOT/loop_tool_service/benchmarks/mm128_16_8_128/0123.txt
```

Format Matrix Multiply: 
- mm{loop0_size}\_{loop1_size}...\_{loopN_size}

### mm128_128_128
```
for m_8757 in 128  
 for k_8758 in 128
  for n_8796 in 128
   %2[m_8757, k_8758, n_8796] <- multiply(%0, %1)
   %3[m_8757, n_8796] <- add(%2)
 for n_8796 in 128
  %4[m_8757, n_8796] <- write(%3)
```

### mm8_16_128_128
```
for m_8757 in 8
 for m_8757 in 16
  for k_8758 in 128
   for n_8796 in 128 
    %2[m_8757, k_8758, n_8796] <- multiply(%0, %1)
    %3[m_8757, n_8796] <- add(%2)
  for n_8796 in 128
   %4[m_8757, n_8796] <- write(%3)
```

### mm8_16_8_16_8_16
```
for m_8757 in 8 
 for m_8757 in 16
  for k_8758 in 8
   for k_8758 in 16
    for n_8796 in 8
     for n_8796 in 16
      %2[m_8757, k_8758, n_8796] <- multiply(%0, %1)
      %3[m_8757, n_8796] <- add(%2)
  for n_8796 in 128
   %4[m_8757, n_8796] <- write(%3)
```



## Training with RLlib

Once we are created benchmark dataset we are ready to train agent to optimize it with RLlib. For the logging of the training we use Weights and Biases and to make program working copy your wandb key in
wandb_key.txt folder in LOOP_TOOL_ROOT directory.

To launch traning locally run:

```
cd loop_tool_service/models/rllib
python rllib_agent.py --iter=10 --dataset=mm128_16_8_128
```

Additionally, you can retrain the model by calling
```
cd loop_tool_service/models/rllib
python rllib_agent.py --iter=10 --dataset=mm128_16_8_128 --wandb_url=$wandb_run_path
```

To run this on SLURM use:
```
python launcher/slurm_launch.py --app=rllib_agent.py --time=5:00 -nc=80 -ng=2 --iter=10 --sweep --dataset=mm128_8_16_128
```
All results from the experiment will be logged on Wandb permanently and localy at:
$LOOP_TOOL_ROOT/loop_tool_service/models/rllib/my_artifacts 

For SLURM output check:
$LOOP_TOOL_ROOT/loop_tool_service/models/rllib/results 

```python




1. for i in 256: <<<<<<<<< agent
2.  for j in 256:
3.   for k in 256:
4.    T0[i, j, k] <- multiply(A, B) 
5.    T1[i, k] <- add(T0) 
6.  for k in 256:
7.   C[i, k] <- write(T1)


1. for i in 16: <<<<<<<<< agent
2.  for i in 16:
3.   for j in 256:
4.    for k in 256:
5.     T0[i, j, k] <- multiply(A, B) 
6.     T1[i, k] <- add(T0) 
7.   for k in 256:
8.    C[i, k] <- write(T1)


1. for i in 16:
2.  for j in 16: <<<<<<<<< agent
3.   for i in 16:
4.    for j in 4:
5.     for k in 16:
6.      for j in 4:
7.       for k in 16:
8.        T0[i, j, k] <- multiply(A, B)  
9.        T1[i, k] <- add(T0)  
10. for i in 16: 
11.  for k in 256:
12.   C[i, k] <- write(T1)  


for i in 256 : L0 
 for j in 256 : L1 <<<<<<<<< agent 
  for k in 256 : L2 
   T0[i, j, k] <- multiply(A, B) 
   T1[i, k] <- add(T0) 
 for k in 256 : L5 
  C[i, k] <- write(T1)


for i in 256 : L0 
 for j in 16 : L1 <<<<<<<<< agent 
  for j in 16 : L1
   for k in 256 : L2 
    T0[i, j, k] <- multiply(A, B) 
    T1[i, k] <- add(T0) 
  for k in 256 : L5 
   C[i, k] <- write(T1)




20.14 GFLOPS
1. for i in 112: <<<<<< agent
2.  for k in 192: 
3.   for j in 128: 
4.    T0[i, k, j] <- multiply(A, B)  
5.    T1[i, j] <- add(T0)  
6.  for j in 128:
7.   C[i, j] <- write(T1) 


['split_4']
20.96 GFLOPS
1. for i in 28: <<<<<< agent
2.  for i in 4:
3.   for k in 192:
4.    for j in 128:
5.     T0[i, k, j] <- multiply(A, B)
6.     T1[i, j] <- add(T0)
7.   for j in 128:
8.    C[i, j] <- write(T1)


['split_4', 'swap_down', 'swap_down', 'down', 'split_16', 'swap_up', 'swap_up']
78.51 GFLOPS
1. for i in 28:
2.  for j in 8: <<<<<< agent
3.   for k in 192:
4.    for i in 4:
5.     for j in 16:
6.      T0[i, k, j] <- multiply(A, B)
7.      T1[i, j] <- add(T0)
8.  for i in 4:
9.   for j in 128:
10.   C[i, j] <- write(T1)
  
  ```

## Evaluating the search

```
python rllib_agent.py --iter=0 --dataset=mm64_256_16_range  --wandb_url=dejang/loop_tool_agent_split/3be38_00000 --trainer=dqn.ApexTrainer --eval_size=50 --eval_time=60
```


```

```


digraph G {
 node [fontname = "courier", fontsize=12];
L0 [shape=record,label="for i in 256 r 0",feature_dict="{'cursor':1,'size':256,'tail':0,'type':1,'unroll':0,'vectorize':0,}"];
L0 -> L1 [color="black",label="",feature_dict="{'type':3,}"];
L0 -> L5 [color="black",label="",feature_dict="{'type':3,}"];
L1 [shape=record,label="for j in 256 r 0",feature_dict="{'cursor':0,'size':256,'tail':0,'type':1,'unroll':0,'vectorize':0,}"];
L1 -> L2 [color="black",label="",feature_dict="{'type':3,}"];
L2 [shape=record,label="for k in 128 r 0",feature_dict="{'cursor':0,'size':128,'tail':0,'type':1,'unroll':0,'vectorize':0,}"];
L2 -> L3 [color="black",label="",feature_dict="{'type':3,}"];
L2 -> L4 [color="black",label="",feature_dict="{'type':3,}"];
L3 [shape=hexagon,label="multiply(A, B)",feature_dict="{'cursor':0,'type':0,}"];
D0 [shape=ellipse,label="A[i, j]",feature_dict="{'type':2,}"];
L0 -> D0 [color="red",label="256",feature_dict="{'stride':1,'type':4,}"];
L1 -> D0 [color="red",label="1",feature_dict="{'stride':1,'type':4,}"];
L2 -> D0 [color="red",label="0",feature_dict="{'stride':1,'type':4,}"];
D0 -> L3 [color="blue",label="",feature_dict="{'stride':1,'type':4,}"];
D1 [shape=ellipse,label="B[j, k]",feature_dict="{'type':2,}"];
L0 -> D1 [color="red",label="0",feature_dict="{'stride':1,'type':4,}"];
L1 -> D1 [color="red",label="128",feature_dict="{'stride':1,'type':4,}"];
L2 -> D1 [color="red",label="1",feature_dict="{'stride':1,'type':4,}"];
D1 -> L3 [color="blue",label="",feature_dict="{'stride':1,'type':4,}"];
D2 [shape=ellipse,label="T[i, j, k]",feature_dict="{'type':2,}"];
L3 -> D2 [color="blue",label="",feature_dict="{'stride':1,'type':4,}"];
L4 [shape=hexagon,label="add<j>(T)",feature_dict="{'cursor':0,'type':0,}"];
D2 -> L4 [color="blue",label="",feature_dict="{'stride':1,'type':4,}"];
D3 [shape=ellipse,label="T'[i, k]",feature_dict="{'type':2,}"];
L1 -> D3 [color="red",label="0",feature_dict="{'stride':1,'type':4,}"];
L2 -> D3 [color="red",label="1",feature_dict="{'stride':1,'type':4,}"];
L4 -> D3 [color="blue",label="",feature_dict="{'stride':1,'type':4,}"];
L5 [shape=record,label="for k in 128 r 0",feature_dict="{'cursor':0,'size':128,'tail':0,'type':1,'unroll':0,'vectorize':0,}"];
L5 -> L6 [color="black",label="",feature_dict="{'type':3,}"];
L6 [shape=hexagon,label="%write(T')",feature_dict="{'cursor':0,'type':0,}"];
D3 -> L6 [color="blue",label="",feature_dict="{'stride':1,'type':4,}"];
D4 [shape=ellipse,label="C[i, k]",feature_dict="{'type':2,}"];
L0 -> D4 [color="red",label="128",feature_dict="{'stride':1,'type':4,}"];
L5 -> D4 [color="red",label="1",feature_dict="{'stride':1,'type':4,}"];
L6 -> D4 [color="blue",label="",feature_dict="{'stride':1,'type':4,}"];
}