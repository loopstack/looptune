from loop_tool_service.models.evaluator import Evaluator
from loop_tool_service.models.rllib.rllib_agent import Agent
from loop_tool_service.models.rllib.rllib_agent import RLlibAgent

from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

from loop_tool_service.paths import LOOP_TOOL_ROOT
import os





parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--wandb_url",  type=str, nargs='?', default='', help="Wandb uri to load policy network."
)
parser.add_argument(
    "--sweep",  type=int, nargs='?', const=1, default=0, help="Run with wandb sweeps."
)
parser.add_argument(
    "--slurm", 
    default=False, 
    action="store_true",
    help="Run on slurm."
)
parser.add_argument(
    "--iter", type=int, default=2, help="Number of iterations to train."
)

parser.add_argument(
    "--dataset",  type=str, nargs='?', help="Dataset [mm128_128_128] to run must be defined in loop_tool_service.service_py.datasets."
)

parser.add_argument(
    "--size", type=int, nargs='?', default=1000000, help="Size of benchmarks to evaluate."
)




class Arena():
    # 1. Create all environments

    # 2. Create all agent and associate them with environment

    # 3. For Each (env, agent) Launch 1 round of training on SLURM

    # 4. Evaluate agents on all environments and associate the best on each env

    # 5. Tweek training parameters for each agent

    # 6. GOTO 3

    def __init__(self, datasets, max_episode_steps, train_iter):
        self.max_episode_steps = max_episode_steps
        self.datasets = datasets
        self.env_agent_map = {} # key: env, value: agent
        self.train_iter = train_iter
        self.results = {} 


    def reset_environments(self):
        self.env_agent_map = {}
        for dataset in self.datasets:
            self.env_agent_map[dataset] = RLlibAgent(algorithm=PPOTrainer, dataset=dataset)


    def train(self):
        
        for env, agent in self.env_agent_map.items():
            self.train_agent(agent)

        
        for env in self.env_agent_map.keys(): 
            results = {}
            for agent in self.env_agent_map.values():        
                results[agent] = self.evaluate(agent, env)
        
            self.env_agent_map[env] = max(results, key=results.get)
                
    

  

    def train_agent(self, agent):
    '''
        Submit SLURM job to train the agent
    '''
        if wandb_url:
            agent.load_model(wandb_url)

        models = agent.train(
            config=config, 
            train_iter=iter, 
            sweep_count=sweep_count
        )


    def evaluate_agent(self, agent, env):
        evaluator = Evaluator(steps=self.max_episode_steps, cost_path="", policy_path=policy_model['policy_path'])
        pass

    

    
if __name__ == '__main__':
    args = parser.parse_args()
    arena = Arena(
        datasets=args.datasets.split(','), 
        max_episode_steps=20, 
        train_iter=10,
    )

    arena.reset_environments()

    arena.train()