import wandb
import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import random
import numpy as np
from loop_tool_service.service_py.utils import timed_fn
import torch
import tempfile
from pathlib import Path
from itertools import cycle, islice

class Evaluator:
    """ Evaluator runs specified searches on full dataset or single benchmark 
    and plots the graphs. This includes greedy, beam searches with loop_nest
    evaluation, as well as searches using policy and cost models.
    """

    #############################################################
    # Public
    #############################################################
    def __init__(self, steps=10, cost_path='', policy_path='', agent=None, reward="flops_loop_nest_cached", timeout=10, debug=False):
        self.set_cost_path(cost_path)
        self.set_policy_path(policy_path)
        self.agent = agent
        self.steps = steps
        self.reward = reward
        self.timeout = timeout
        self.debug = "--debug" if debug else ""
        self.searches = {
            'greedy1_ln': f'greedy_search --steps={self.steps} --lookahead=1 --width=1000 --eval=loop_nest --timeout={self.timeout} {self.debug}',
            'greedy2_ln': f'greedy_search --steps={self.steps} --lookahead=2 --width=1000 --eval=loop_nest --timeout={self.timeout} {self.debug}',
            'beam2dfs_ln': f'beam_search_dfs --steps={self.steps} --width=2 --ranking --eval=loop_nest --timeout={self.timeout} {self.debug}',           
            'beam4dfs_ln': f'beam_search_dfs --steps={self.steps} --width=4 --ranking --eval=loop_nest --timeout={self.timeout} {self.debug}',     
            'beam2bfs_ln': f'beam_search_bfs --steps={self.steps} --width=2 --ranking --eval=loop_nest --timeout={self.timeout} {self.debug}',                 
            'beam4bfs_ln': f'beam_search_bfs --steps={self.steps} --width=4 --ranking --eval=loop_nest --timeout={self.timeout} {self.debug}',                 
            'bruteforce_ln': f'beam_search_dfs --steps={self.steps} --width=1000 --eval=loop_nest --timeout={self.timeout} {self.debug}',
            'random_ln': f'random_search --steps={self.steps} --eval=loop_nest --timeout={self.timeout} {self.debug}',           

            'greedy1_cost': f'greedy_search --steps={self.steps} --lookahead=1 --width=1000 --eval=cost --timeout={self.timeout} {self.debug}',
            'greedy2_cost': f'greedy_search --steps={self.steps} --lookahead=2 --width=1000 --eval=cost --timeout={self.timeout} {self.debug}',
            'bruteforce_cost': f'beam_search_dfs --steps={self.steps} --width=1000 --eval=cost --timeout={self.timeout} {self.debug}',

            'loop_tune_ln': f'greedy_search --steps={self.steps} --lookahead=1 --width=1000 --eval=policy --timeout={self.timeout} {self.debug}',

        }
        self.my_artifacts = Path(tempfile.mkdtemp()) # Dir to download and upload files. Has start, end subdirectories



        
    def set_cost_path(self, path):
        self.cost_path = str(path)

    def set_policy_path(self, path):
        self.policy_path = str(path)

    def set_policy_agent(self, agent):
        self.agent = agent

    def evaluate(self, env, benchmarks: list, searches: list):
        """ Run run and plot searches on benchmarks

        Args:
            env (CompilerGymEnv): made environment
            benchmarks (list): list of string names of benchmarks to evaluate
            searches (list): liset [search_name]. Check handle_session_parameter for format
        """
        
        self.df_gflops, self.df_time, self.df_actions = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for benchmark in tqdm(sorted(benchmarks)):
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
            searches (list): [search_name]. Check handle_session_parameter for format

        Returns:
            dict, dict, dict: gflops, time, actions dict for each search
        """
        searches_cmd = { k:v for k, v in self.searches.items() if k in searches} 

        benchmark_name = str(benchmark).split('/')[-1]
        results_gflops = {'benchmark': benchmark_name}
        results_time = {'benchmark': benchmark_name}
        results_actions = {'benchmark': benchmark_name}

        env.reset(benchmark=benchmark)
        env.send_param('load_cost_model', self.cost_path)
        env.send_param('load_policy_model', self.policy_path)  
        
        results_actions['base'], results_gflops['base'], results_time['base'] = self.base_performance(env, eval_mode='loop_nest')

        print(benchmark)
        env.send_param("print_looptree", "")
        print(f"Base performance = {results_gflops['base']}")

        for search_name, search_cmd in tqdm(searches_cmd.items()):
            if search_name == 'loop_tune_ln' and self.agent != None:
                results_actions[search_name], results_gflops[search_name], results_time[search_name] = self.rllib_search(env, results_gflops['base'][0])
            else:    
                results_actions[search_name], results_gflops[search_name], results_time[search_name]  = self.search_performance(env, search_cmd, results_gflops['base'])
        
        return results_gflops, results_time, results_actions


    def save(self, path):
        bench_column, search_columns = self.df_gflops.columns[0], self.df_gflops.columns[1:]
        df_gflops_final = self.df_gflops[search_columns].apply(lambda x: x.str[-1]) 
        df_gflops_final.insert(loc=0, column=bench_column, value=self.df_gflops[bench_column])
        df_time_final = self.df_time[search_columns].apply(lambda x: x.str[-1]) 
        df_time_final.insert(loc=0, column=bench_column, value=self.df_time[bench_column])

        self.plot_bars(df_gflops_final, df_time_final, path)
        self.plot_violin(df_gflops_final, path)
        self.plot_actions(path)
        df_all = pd.concat( [self.df_gflops, self.df_time, self.df_actions], axis=1)
        df_all.to_csv(f'{path}.csv')

    
        # for _, row in df_all.iterrows():
        #     print(f"\n_______________________________________________________________")
        #     for x in row: print(x)

       

    def plot_bars(self, df_gflops_final, df_time_final, path):
        if type(df_gflops_final) == None and type(df_time_final) == None:
            return
        fig, axs = plt.subplots(2, 1)
        num_bench = min(len(df_gflops_final), 100)
        figsize = ((num_bench + 1) // 2, 5)
        indexes = sorted(random.sample(range(len(df_gflops_final)), num_bench))

        bench_column, search_columns = df_gflops_final.columns[0], df_gflops_final.columns[1:]

        axs[0] = df_gflops_final.iloc[indexes].plot(x=bench_column, y=search_columns, kind='bar', figsize=figsize, width=0.8, align='edge', ax=axs[0])
        axs[0].minorticks_on()
        axs[0].grid(which='both', axis='y')
        axs[0].set_ylabel('GFLOPS')
        axs[0].legend(title='Searches',loc='center left', bbox_to_anchor=(1, 0.5))

        if type(df_time_final) != None:
            axs[1] = df_time_final.iloc[indexes].plot(x=bench_column, y=search_columns, kind='bar', figsize=figsize, width=0.8, align='edge', ax=axs[1])
            # axs[1].minorticks_on()
            axs[1].grid(which='major', axis='y')
            axs[1].set_ylabel('seconds')
            axs[1].set_yscale('log')
            axs[1].get_legend().remove()


        plt.gca().yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
        # plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))


        fig.suptitle(f'Benchmarks evaluation', fontsize=16)
        fig.autofmt_xdate()
        fig.savefig(f'{path}_bars.png', bbox_inches = 'tight')


    def plot_violin(self, df_gflops_final, path):
        if type(df_gflops_final) == None: 
            return

        # Analyse results
        fig, axs = plt.subplots()
        labels = df_gflops_final.columns[2:]

        # valid_gflops = df_gflops_final[(df_gflops_final != 0).all(1)] # Filter all rows where gflops are 0
        
        axs.violinplot(
            dataset = [ 
                df_gflops_final[col].astype(float) / df_gflops_final['base'].astype(float) 
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
        axs.tick_params(labelrotation=0)


    def plot_actions(self, path):
        if type(self.df_gflops) == None: 
            return

        bench_column, search_columns = self.df_gflops.columns[0], self.df_gflops.columns[1:]

        # for best_idx in range(len(df_gflops_final)): #df_gflops_final[search_columns].idxmax():
        for index in range(len(self.df_gflops)):
            gflops_row = self.df_gflops.iloc[index]
            time_row = self.df_time.iloc[index]

            fig, axs = plt.subplots(2, 1)
            
            for i, search in enumerate(search_columns): 
                x_data = np.arange(0, 2 * self.steps, 1) + 0.05 * i  # to show overlaping points
                axs[0].plot(x_data[:len(gflops_row[search])], gflops_row[search], marker = '.', label = search)      
                axs[1].plot(x_data[:len(time_row[search])], time_row[search], marker = '.', label = search)


            fig.suptitle(f'Gain per action', fontsize=16)
            axs[0].legend(title='Searches',loc='center left', bbox_to_anchor=(1, 0.5))
            axs[0].set_xticks(range(self.steps + 1))
            axs[0].set_ylabel('GFLOPS')
            axs[0].grid(b=True, which='major', axis='y')

            axs[1].set_xlabel('steps')
            axs[1].set_xticks(range(self.steps + 1))
            axs[1].set_ylabel('seconds')
            axs[1].set_yscale('log')
            axs[1].grid(b=True, which='major', axis='y', linestyle='-')
            axs[1].grid(b=True, which='minor', axis='y', linestyle='--')
            
            plt.gca().yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
            plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))

            fig.savefig(f'{path}_{gflops_row[bench_column]}_actions.png', bbox_inches = 'tight')
            print(f'\n{path}_{gflops_row[bench_column]}_actions.png\n')
        

    def send_to_wandb(self, wandb_run_id, wandb_dict=None, path=None):
        if wandb_run_id == 'dummy': 
            return

        wandb_dict['group_id'] = wandb_run_id.split('_')[0]
        wandb_dict['run_id'] = wandb_run_id

        wandb_url = f'dejang/loop_tool_agent_split/{wandb_run_id}'
        api = wandb.Api()
        wandb_run = api.run(wandb_url)

        if wandb_dict: # Upload wandb dict
            for key, value in wandb_dict.items(): 
                wandb_run.summary[key] = value
            wandb_run.summary.update()

        # Delete previous files 
        for file in wandb_run.files():
            if file.name.endswith('.csv') or file.name.endswith('.png'): 
                file.delete()

        if path: # Upload wandb plots
            cwd = os.getcwd()
            os.chdir(path)
            for root, dirs, files in os.walk(path):
                for file in files:
                    print(f"{root}/{file}")
                    wandb_run.upload_file(f"{root}/{file}")       
            os.chdir(cwd)
        
        print(f'\nWandb page = https://wandb.ai/{wandb_url}')

    #############################################################
    # Private
    #############################################################
    def base_performance(self, env, eval_mode='loop_nest'):
        start = time.time()
        if eval_mode == 'loop_nest':
            gflops = float(env.observation[self.reward])
        elif eval_mode == 'cost':
            gflops = float(env.observation['gflops_cost'])
        else:
            assert(0), 'base performance eval_mode must be loop_nest or cost'
        return [], [gflops], [time.time() - start]


    def search_performance(self, env, search, base_gflops):
        search_cmd, search_args = search.split(" ", 1)
        env.send_param("reset_agent", '')
        start = time.time()
        res = env.send_param(search_cmd, search_args)
        search_time = time.time() - start
        if res != None:
            search_table = json.loads(res)
            actions, action_gflops, action_times = search_table #list(zip(*search_table)) # unzip search table
            # gflops = self.move_and_eval(env, actions_str=actions)
        else:
            actions, action_gflops, action_times = ["failed"], base_gflops, [search_time]

        return actions, action_gflops, action_times #gflops, search_time, actions


    def rllib_search(self, env, base_gflops):
        actions, action_gflops, action_times = [''], [base_gflops], [float('nan')]
        start_time = time.time()
        feature_vector = env.observation["loops_tensor"]

        for i in range(self.steps):
            feature_vector = torch.Tensor(feature_vector).unsqueeze(0)
            a_id = self.agent.compute_action(feature_vector)
            action = env.action_space.to_string(a_id)
            actions.append(action)
            action_gflops.append(float(env.observation[self.reward]))
            action_times.append(time.time() - start_time)
            env.step(a_id)
            feature_vector = env.observation["loops_tensor"]

        return actions, action_gflops, action_times


    def move_and_eval(self, env, actions_str):
        actions_ids = [ env.action_space.from_string(a) for a in actions_str ]
        env.multistep(actions_ids)
        return float(env.observation[self.reward])
