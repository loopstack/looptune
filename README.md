# LoopTune project

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
# How to run

## 0. Setup environment variables:
```
export LOOP_TOOL_ROOT=$path_to_this_dir
export WANDB_CONSOLE=off
export MAX_GFLOPS=$peak_gflops_measurement
export RLLIB_NUM_GPUS=$num_gpus_available
```
### Enable Wandb logging
Create Weight and Biases account and put your wandb key in $LOOP_TOOL_ROOT/wandb_key.txt


## 1. Generate dateset:
```
python loop_tool_service/benchmarks/generator.py --kind=mm --dimA=64:128:16,64:128:16 --dimB=64:128:16,64:128:16  --out=$LOOP_TOOL_ROOT/loop_tool_service/benchmarks/mm64_128_16_range
# Register dataset
python setup.py install
```

To examine any created benchmark just run:
```
python loop_tool_service/benchmarks/reader.py --bench=path-to-txt-benchmark --actions='csv-actions'
```


## 2. Train policy model:

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

Additionally, you can start training from finished training logged on wandb with adding
```
--wandb_url=$wandb_run_path
```
to the training commands.



## 4. Evaluation:

### Comparison to traditional searches
Once the training is done you can reproduce evaluation with:
```
cd loop_tool_service/experiments/compare_searches
python search.py --trainer=dqn.ApexTrainer --wandb_url=$wandb_run_path
```

If you copy graph to online graphviz you can see visualization of 
searches by adding --debug.


## Comparison to Numpy and TVM

```
cd loop_tool_service/experiments/compare_tvm
python compare_tvm.py --size=25 --wandb_url=$wandb_run_path
```