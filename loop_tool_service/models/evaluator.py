import wandb
import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import random
import numpy as np

class Evaluator:
    """ Evaluator runs specified searches on full dataset or single benchmark 
    and plots the graphs. This includes greedy, beam searches with loop_nest
    evaluation, as well as searches using policy and cost models.
    """

    #############################################################
    # Public
    #############################################################
    def __init__(self, steps=10, cost_path='', policy_path=''):
        self.cost_path = cost_path
        self.policy_path = policy_path
        self.steps = steps
        
        self.searches = {
            'greedy1_ln': f'greedy_search --steps={self.steps} --lookahead=1 --width=1000 --eval=loop_nest',
            'greedy1_cost': f'greedy_search --steps={self.steps} --lookahead=1 --width=1000 --eval=cost',
            'greedy1_policy': f'greedy_search --steps={self.steps} --lookahead=1 --width=1000 --eval=policy',
            'greedy2_ln': f'greedy_search --steps={self.steps} --lookahead=2 --width=1000 --eval=loop_nest',
            'greedy2_cost': f'greedy_search --steps={self.steps} --lookahead=2 --width=1000 --eval=cost',
            'greedy2_policy': f'greedy_search --steps={self.steps} --lookahead=2 --width=1000 --eval=policy',            
            'bruteforce_ln': f'beam_search --steps={self.steps} --width=1000 --eval=loop_nest',
            'bruteforce_cost': f'beam_search --steps={self.steps} --width=1000 --eval=cost',
            'bruteforce_policy': f'beam_search --steps={self.steps} --width=1000 --eval=policy',
            'policy_ln': f'beambeam_search --steps1={self.steps//2} --width1=2 --eval1=policy --steps2={self.steps - self.steps//2} --width2=2 --eval2=loop_nest',
            'policy_cost': f'beambeam_search --steps1={self.steps//2} --width1=2 --eval1=policy --steps2={self.steps - self.steps//2} --width2=2 --eval2=cost',
        }
        
    def evaluate(self, env, benchmarks: list, searches: dict):
        """ Run run and plot searches on benchmarks

        Args:
            env (CompilerGymEnv): made environment
            benchmarks (list): list of string names of benchmarks to evaluate
            searches (dict): dict {search_name: search_cmd}. Check handle_session_parameter for format
        """
        self.df_gflops, self.df_time, self.df_actions = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for benchmark in tqdm(benchmarks):
            results_gflops, results_time, results_actions = self.evaluate_single_benchmark(env, benchmark, searches)
            self.df_gflops = pd.concat([self.df_gflops, pd.DataFrame([results_gflops])], axis=0)
            self.df_time = pd.concat([self.df_time, pd.DataFrame([results_time])], axis=0)
            self.df_actions = pd.concat([self.df_actions, pd.DataFrame([results_actions])], axis=0)

        return { 'gflops': self.df_gflops, 'time': self.df_time, 'actions': self.df_actions }
        

    def evaluate_single_benchmark(self, env, benchmark, searches):
        """ Run set of searches on single benchmark

        Args:
            env (CompilerGymEnv): environment.
            benchmark (str): benchmark to run.
            searches (dict): {search_name: search_cmd}. Check handle_session_parameter for format

        Returns:
            dict, dict, dict: gflops, time, actions dict for each search
        """
        results_gflops = {'benchmark': benchmark}
        results_time = {'benchmark': benchmark}
        results_actions = {'benchmark': benchmark}

        env.reset(benchmark=benchmark)
        env.send_param('load_cost_model', str(self.cost_path))
        env.send_param('load_policy_model', str(self.policy_path))        
        
        results_gflops['base'], results_time['base'], results_actions['base'] = self.base_performance(env, eval_mode='loop_nest')

        print(benchmark)
        env.send_param("print_looptree", "")
        print(f"Base performance = {results_gflops['base']}")

        for search_name, search_cmd in searches.items():
            results_gflops[search_name], results_time[search_name], results_actions[search_name] = self.search_performance(env, search_cmd)
            
        return results_gflops, results_time, results_actions


    def save(self, path):
        self.plot_bars(path)
        self.plot_violin(path)
        pd.concat( [self.df_gflops, self.df_time, self.df_actions], axis=1).to_csv(f'{path}.csv')

        for _, row in self.df_actions.iterrows():
            print(f"\n_______________________________________________________________")
            for search, actions in row.items():
                print(f"\t{search}, {actions}")
       

    def plot_bars(self, path):
        fig, axs = plt.subplots(2, 1)
        num_bench = min(len(self.df_gflops), 100)
        figsize = ((num_bench + 1) // 2, 5)
        indexes = sorted(random.sample(range(len(self.df_gflops)), num_bench))

        axs[0] = self.df_gflops.iloc[indexes].plot(x=self.df_gflops.columns[0], y=self.df_gflops.columns[1:], kind='bar', figsize=figsize, width=0.8, align='edge', ax=axs[0])
        axs[0].minorticks_on()
        axs[0].grid(which='both', axis='y')
        axs[0].set_ylabel('GFLOPS')
        
        axs[1] = self.df_time.iloc[indexes].plot(x=self.df_time.columns[0], y=self.df_time.columns[1:], kind='bar', figsize=figsize, width=0.8, align='edge', ax=axs[1])
        axs[1].minorticks_on()
        axs[1].grid(which='both', axis='y')
        axs[1].set_ylabel('seconds')
        axs[1].set_yscale('log')
        
        fig.suptitle(f'Benchmarks evaluation', fontsize=16)
        fig.autofmt_xdate()

        axs[0].legend(title='Searches',loc='center left', bbox_to_anchor=(1, 0.5))
        axs[1].get_legend().remove()

        fig.savefig(f'{path}_bars.png', bbox_inches = 'tight')


    def plot_violin(self, path):
        # Analyse results
        fig, axs = plt.subplots()
        labels = self.df_gflops.columns[2:]
        axs.violinplot(
            dataset = [ 
                self.df_gflops[col].astype(float) / self.df_gflops['base'].astype(float) 
                for col in labels # no benchmark, base columns
            ],
            showmedians=True
        )
        axs.set_xticks(np.arange(1, len(labels) + 1))
        axs.set_xticklabels(labels)
        axs.set_xlim(0.25, len(labels) + 0.75)
        axs.tick_params(labelrotation=45)

        axs.set_title('Speedup distribution')
        axs.yaxis.grid(True)
        axs.set_xlabel('Models')
        fig.savefig(f"{path}_violin.png", bbox_inches = 'tight')



    def send_to_wandb(self, path, wandb_run_id, wandb_dict):
        wandb_dict['group_id'] = wandb_run_id.split('_')[0]
        wandb_dict['run_id'] = wandb_run_id

        cwd = os.getcwd()
        os.chdir(path)
        wandb_uri = f'dejang/loop_tool_agent_split/{wandb_run_id}'
        print(f'Wandb page = https://wandb.ai/{wandb_uri}')
        api = wandb.Api()
        wandb_run = api.run(wandb_uri)


        for root, dirs, files in os.walk(path):
            for file in files:
                print(f"{root}/{file}")
                wandb_run.upload_file(f"{root}/{file}")        

        for key, value in wandb_dict.items(): 
            wandb_run.summary[key] = value
        wandb_run.summary.update()
        os.chdir(cwd)

    #############################################################
    # Private
    #############################################################
    def base_performance(self, env, eval_mode='loop_nest'):
        start = time.time()
        if eval_mode == 'loop_nest':
            gflops = env.observation['flops_loop_nest']
        elif eval_mode == 'cost':
            gflops = env.observation['gflops_cost']
        else:
            assert(0), 'base performance eval_mode must be loop_nest or cost'
        return gflops, time.time() - start, ""


    def search_performance(self, env, search):
        search_cmd, search_args = search.split(" ", 1)
        env.send_param("reset_agent", '')
        try:
            start = time.time()
            actions_reward = json.loads(env.send_param(search_cmd, search_args))
            search_time = time.time() - start
            gflops = self.move_and_eval(env, actions_str=actions_reward[0])
        except TimeoutError:
            gflops, search_time = 0, 0
            actions_reward = "failed"

        return gflops, search_time, actions_reward[0]


    def move_and_eval(self, env, actions_str):
        actions_ids = [ env.action_space.from_string(a) for a in actions_str ]
        env.multistep(actions_ids)
        return env.observation['flops_loop_nest']
