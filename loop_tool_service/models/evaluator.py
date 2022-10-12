import json
import pandas as pd
from matplotlib import pyplot as plt
import time
from tqdm import tqdm


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
        df_gflops, df_time = None, None
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
        self.df_gflops, self.df_time = pd.DataFrame(), pd.DataFrame()
        for i, benchmark in tqdm(enumerate(benchmarks)):
            results_gflops, results_time = self.evaluate_single_benchmark(env, benchmark, searches)
            self.df_gflops = pd.concat([self.df_gflops, pd.DataFrame([results_gflops])], axis=0)
            self.df_time = pd.concat([self.df_time, pd.DataFrame([results_time])], axis=0)


        return self.df_gflops, self.df_time
        

    def save(self, path, yscale='linear'):
        fig, axs = plt.subplots(2, 1)
        breakpoint()
        axs[0] = self.df_gflops.plot(x=self.df_gflops.columns[0], y=self.df_gflops.columns[1:], kind='bar', figsize=(40, 5), ax=axs[0])
        axs[0].minorticks_on()
        axs[0].grid(which='both', axis='y')
        axs[0].set_ylabel('GFLOPS')
        
        axs[1] = self.df_time.plot(x=self.df_time.columns[0], y=self.df_time.columns[1:], kind='bar', figsize=(40, 5), ax=axs[1])
        axs[1].minorticks_on()
        axs[1].grid(which='both', axis='y')
        axs[1].set_ylabel('seconds')
        

        fig.suptitle(f'GFlops comparison benchmarks', fontsize=16)
        fig.autofmt_xdate()
        # fig.tight_layout()

        axs[0].legend(title='SUBJECT',loc='center left', bbox_to_anchor=(1, 0.5))
        axs[1].get_legend().remove()

        fig.savefig(f'{path}/results.png', bbox_inches = 'tight')
        breakpoint()
        pd.concat( [self.df_gflops, self.df_time], axis=1).to_csv(f'{path}/results.csv')
       

    def evaluate_single_benchmark(self, env, benchmark, searches):
        results_gflops = {'benchmark': benchmark}
        results_time = {'benchmark': benchmark}
        env.reset(benchmark=benchmark)
        env.send_param("print_looptree", "")
        env.send_param('load_cost_model', str(self.cost_path))
        env.send_param('load_policy_model', str(self.policy_path))        
        
        results_gflops['base'], results_time['base'] = self.base_performance(env, eval_mode='loop_nest')
        for search_name, search_cmd in searches.items():
            results_gflops[search_name], results_time[search_name] = self.search_performance(env, search_cmd)
            
        return results_gflops, results_time


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
        return gflops, time.time() - start


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

        return gflops, search_time


    def move_and_eval(self, env, actions_str):
        actions_ids = [ env.action_space.from_string(a) for a in actions_str ]
        env.multistep(actions_ids)
        return env.observation['flops_loop_nest']

 