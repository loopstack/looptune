import torch 
import numpy as np
import loop_tool as lt
import time

from compiler_gym.service.proto import (
    Event,
    ByteTensor,
    FloatTensor,
)

class Evaluator:
    def __init__(self, env):
        self.env = env
        self.cost_model = None
        self.policy_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ln_cache = {}
        self.cost_cache = {}
        self.policy_cache = {}
        self.time_spent = []
        self.inputs = []

    def load_cost_model(self, model_path_str):
        if model_path_str == '':
            self.cost_model = None
        else:
            # self.cost_model = torch.load(model_path_str)
            del self.cost_model
            self.cost_model = torch.jit.load(model_path_str).to(self.device)
            self.cost_model.eval()
            self.reset_cuda(self.env.agent)


    def reset_cuda(self, agent):
        '''
        This will create computation graph for pytorch cuda
        '''
        state_tensor = torch.tensor([0] * len(agent.get_stride_histogram())).float().to(self.device)
        self.cost_model(state_tensor)
        del state_tensor
        # torch.cuda.empty_cache()


    def load_policy_model(self, model_path_str):
        if model_path_str == '':
            self.policy_model = None
        else:
            self.policy_model = torch.load(model_path_str)
            self.policy_model.eval()


    def get_actions_q_cost(self, agent, eval_mode)->dict:
        actions_q = {}
        for action in agent.get_available_actions():
            agent_copy = agent.copy()
            agent_copy.apply_action(action)
            actions_q[action] = self.eval_gflops(agent_copy, eval_mode=eval_mode)
        
        # breakpoint()
        return sorted(actions_q.items(), key=lambda x: x[0], reverse=True)


    def eval_gflops(self, agent, eval_mode):
        tree_hash = hash(str(agent.lt))
        if eval_mode == 'cost':
            assert (self.cost_model != None), 'Cost model not loaded'
            if tree_hash in self.cost_cache: 
                return self.cost_cache[tree_hash]
            else:
                start = time.time()
                state_tensor = [ np.log2(x + 1) for x in agent.get_stride_histogram() ]
                state_tensor = torch.tensor(state_tensor).float().to(self.device)
                pred_tensor = self.cost_model(state_tensor)
                pred_gflops = float(pred_tensor.item())
                end = time.time()
                del pred_tensor, state_tensor
                self.time_spent.append(end - start)
                print(f'Costs: \n model = {end - start}')

                print(f"Cost model____________{end - start}_____________{pred_gflops}")

                # aa=time.time();self.cost_model(state_tensor);bb=time.time(); print(bb-aa)
                self.cost_cache[tree_hash] = pred_gflops
                # self.inputs.append(state_tensor)
                return pred_gflops
        else:
            if tree_hash in self.ln_cache: 
                return self.ln_cache[tree_hash]
            else:
                start = time.time()
                gflops = self.env.eval_ln_flops(agent)
                end = time.time()
                self.time_spent.append(end - start)
                self.ln_cache[tree_hash] = gflops
                print(f"LoopNest model______{end - start}_________cache__{len(self.ln_cache)}________{gflops}")
                return gflops


    def get_actions_q_policy_tensor(self, agent):
        feature_vector = self.env.get_loops_tensor(agent=agent).float_tensor.value
        feature_vector = torch.Tensor(feature_vector).unsqueeze(0).to(self.device)
        logits, _ = self.policy_model({"obs": feature_vector}) 
        assert (len(logits.flatten()) == len(agent.action_space)), f"Policy_model_output == {len(logits.flatten())}, while action_space = {agent.action_space}"
        return logits.flatten()
        

    def get_actions_q_policy(self, agent)-> dict:
        print("Policy model_________________________")
        available_actions = agent.get_available_actions()
        logits = self.get_actions_q_policy_tensor(agent)
        actions_q = { self.env.action_space_str[a_id]: float(a_q) for a_id, a_q in enumerate(logits) if self.env.action_space_str[a_id] in available_actions }
        return sorted(actions_q.items(), key=lambda x: x[0], reverse=True)


    def get_actions_q(self, agent, eval_mode):  # mode in ['gflops', 'cost', 'policy'] 
        if eval_mode == 'loop_nest':
            return self.get_actions_q_cost(agent=agent, eval_mode=eval_mode)
        elif eval_mode == 'cost': 
            assert(self.cost_model != None), 'Cost model not loaded'
            return self.get_actions_q_cost(agent=agent, eval_mode=eval_mode)
        elif eval_mode == 'policy': 
            assert(self.policy_model != None), 'Policy model not loaded'
            return self.get_actions_q_policy(agent=agent)
        else:
            assert(0), f"get_actions_q: Not supported mode = {eval_mode}"