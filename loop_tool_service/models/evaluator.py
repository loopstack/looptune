import json
import pandas as pd
from matplotlib import pyplot as plt
import time


class Evaluator:
    """ Evaluator runs specified searches on full dataset or single benchmark 
    and plots the graphs. This includes greedy, beam searches with loop_nest
    evaluation, as well as searches using policy and cost models.
    """

    def __init__(self, cost_path='', policy_path=''):
        self.cost_path = cost_path
        self.policy_path = policy_path

    def load_model(self, env, eval_mode):
        if eval_mode == 'cost':
            env.send_param('load_cost_model', str(self.cost_path))
        if eval_mode == 'policy':
            env.send_param('load_policy_model', str(self.policy_path))


    def eval_benchmark(self, env, benchmark, num_steps, searches):
        print(benchmark)
        # breakpoint()
        results = {}

        for search in searches:
            if search == 'base_ln':
                print('___ reward_base ln___')
                results['reward_base_ln'], results['time_base_ln'] = self.base_performance(env, benchmark, eval_mode='loop_nest')

            elif search == 'base_cost':
                print('___ reward_base cost ___')
                results['reward_base_cost'], results['time_base_cost'] = self.base_performance(env, benchmark, eval_mode='cost')
            
            elif search == 'greedy1_ln':
                print('___ reward_greedy_ln ___')
                results['reward_greedy_ln'], results['time_greedy_ln'] = self.greedy_search(env, benchmark,
                    walk_count=1,
                    step_count=num_steps,
                    search_depth=1,
                    search_width=1000,
                    eval_mode='loop_nest',
                )

            elif search == 'greedy1_cost':
                print('___ reward_greedy_cost ___')
                results['reward_greedy_cost'], results['time_greedy_cost'] = self.greedy_search(env, benchmark,
                    walk_count=1,
                    step_count=num_steps,
                    search_depth=1,
                    search_width=1000,
                    eval_mode='cost',
                )
            elif search == 'greedy2_ln':
                print('___ reward_greedy2_ln ___')
                results['reward_greedy2_ln'], results['time_greedy2_ln'] = self.greedy_search(env, benchmark,
                    walk_count=1,
                    step_count=num_steps,
                    search_depth=2,
                    search_width=1000,
                    eval_mode='loop_nest',
                )
            elif search == 'greedy2_cost':
                print('___ reward_greedy2_cost ___')
                results['reward_greedy2_cost'], results['time_greedy2_cost'] = self.greedy_search(env, benchmark,
                    walk_count=1,
                    step_count=num_steps,
                    search_depth=2,
                    search_width=1000,
                    eval_mode='cost',
                )
            elif search == 'bruteforce_ln':
                print('___ reward_brute_force_ln ___')
                results['reward_brute_force_ln'], results['time_brute_force_ln'] = self.brute_force_search(env, benchmark, 
                    search_width=1000,
                    num_steps=num_steps, 
                    eval_mode='loop_nest'
                )
            elif search == 'bruteforce_cost':
                print('___ reward_brute_force_cost ___')
                results['reward_brute_force_cost'], results['time_brute_force_cost'] = self.brute_force_search(env, benchmark, 
                    search_width=1000,
                    num_steps=num_steps, 
                    eval_mode='cost'
                )
            elif search == 'random':
                print('___ reward_random ___')
                results['reward_random'], results['time_random'] = self.greedy_search(env, benchmark, 
                    walk_count=10,
                    step_count=num_steps,
                    search_depth=0,
                    search_width=1000,
                    eval_mode='loop_nest'
                )
            elif search == 'policy':
                print('__ policy __')
                results['reward_policy'], results['time_policu'] = self.policy_search(env, benchmark, 
                    walk_count=10,
                    step_count=num_steps,
                    search_depth=0,
                    search_width=1000,
                    eval_mode='loop_nest'
                )

            else:
                print(f"Search {search} is not supported! ")
        
        results_gflops = {'benchmark': benchmark}
        results_gflops.update({k:v for k, v in results.items() if k.startswith('reward')})

        results_serch_time = {'benchmark': benchmark}
        results_serch_time.update({k:v for k, v in results.items() if k.startswith('time')})    
        
        self.plot_results([pd.DataFrame([results_gflops]), pd.DataFrame([results_serch_time])], ['linear', 'log'])


    ################# Searches #######################

    def move_and_eval(self, env, actions_str):
        actions_ids = [ env.action_space.from_string(a) for a in actions_str ]
        env.multistep(actions_ids)
        return env.observation['flops_loop_nest']


    def base_performance(self, env, benchmark, eval_mode='loop_nest'):
        env.reset(benchmark=benchmark)
        self.load_model(env, eval_mode)

        env.send_param("print_looptree", "")
        start = time.time()
        if eval_mode == 'loop_nest':
            gflops = env.observation['flops_loop_nest']
        elif eval_mode == 'cost':
            gflops = env.observation['gflops_cost']
        else:
            assert(0), 'base performance eval_mode must be loop_nest or cost'
        search_time = time.time() - start
        print(f'{search_time} Base Search = {gflops}')
        return gflops, search_time

    def brute_force_search(self, env, benchmark, num_steps, search_width, eval_mode='loop_nest'):
        env.reset(benchmark=benchmark)
        self.load_model(env, eval_mode)
        try:
            start = time.time()
            actions_reward = json.loads(env.send_param("beam_search", f'{num_steps}, {search_width}, {eval_mode}'))
            search_time = time.time() - start

            gflops = self.move_and_eval(env, actions_str=actions_reward[0])
            return gflops, search_time
        except TimeoutError:
            gflops, search_time = 0, 0
            actions_reward = "failed"

        print(f'{search_time} Brute Force Search = {actions_reward}')

        return gflops, search_time

    def greedy_search(
        self,
        env,
        benchmark, 
        walk_count,
        step_count,
        search_depth,
        search_width,
        eval_mode,
    ):
        env.reset(benchmark=benchmark)
        self.load_model(env, eval_mode)
        
        try:
            start = time.time()
            actions_reward = json.loads(env.send_param("greedy_search", f'{walk_count}, {step_count}, {search_depth}, {search_width}, {eval_mode}'))
            search_time = time.time() - start
            gflops = self.move_and_eval(env, actions_str=actions_reward[0])
        except TimeoutError:
            gflops, search_time = 0, 0
            actions_reward = "failed"

        print(f'{search_time} Greedy Search = {actions_reward}')
        return gflops, search_time


    def policy_search(
        self,
        env,
        benchmark, 
        walk_count,
        step_count,
        search_depth,
        search_width,
        eval_mode,
    ):
        env.reset(benchmark=benchmark)
        self.load_model(env, eval_mode)
        
        try:
            start = time.time()
            actions_reward = json.loads(env.send_param("policy_search", f'{walk_count}, {step_count}, {search_depth}, {search_width}, {eval_mode}'))
            search_time = time.time() - start
            gflops = self.move_and_eval(env, actions_str=actions_reward[0])
        except TimeoutError:
            gflops, search_time = 0, 0
            actions_reward = "failed"

        print(f'{search_time} Greedy Search = {actions_reward}')
        return gflops, search_time



    def plot_results(save_path, df_gflops_list, yscale):
        if len(df_gflops_list) == 1:
            df_gflops = df_gflops_list[0]
            fig, ax = plt.subplots(1, 1)
            ax = df_gflops.plot(x=df_gflops.columns[0], y=df_gflops.columns[1:], kind='bar', ax=ax)
            ax.minorticks_on()
            ax.grid(which='both', axis='y')
        else:    
            width_ratio = int(len(df_gflops_list[0]) / len(df_gflops_list[1]))
            fig, axs = plt.subplots(1, len(df_gflops_list), figsize=(40, 5), gridspec_kw={'width_ratios': [width_ratio, 1]})
            for i, df_gflops in enumerate(df_gflops_list):
                axs[i] = df_gflops.plot(x=df_gflops.columns[0], y=df_gflops.columns[1:], kind='bar', ax=axs[i])
                axs[i].minorticks_on()
                axs[i].grid(which='both', axis='y')
                axs[i].set_yscale(yscale[i])
        
        fig.suptitle(f'GFlops comparison benchmarks', fontsize=16)
        fig.autofmt_xdate()
        fig.tight_layout()

        fig.savefig(f'{save_path}/results.png')

        pd.concat(df_gflops_list, axis=1).to_csv(f'{save_path}/results.csv')
        