import random

class BeamSearcher:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def search(self, agent, num_steps, search_width, eval_mode, eval_mode2=None): # pair(actions, reward)
        assert (num_steps >= 0)
        if num_steps == 0:
            return [ random.choice(agent.get_available_actions()) ], 0
        else:
            return self.search_n(
                agent=agent, 
                num_steps=num_steps, 
                search_width=search_width,
                n = 1,
                eval_mode=eval_mode,
                eval_mode2=eval_mode2,
            )[0]

    def search_n(self, agent, num_steps, search_width, n, eval_mode, eval_mode2=None):
        agent_copy = agent.copy()
        agent_copy.clear_actions()
        actions_reward = []

        self.beam_search_core(
            agent=agent_copy, 
            search_depth=num_steps, 
            search_width=search_width,
            actions_reward = actions_reward,
            eval_mode=eval_mode,
            eval_mode2=eval_mode2,
        )
        return sorted(actions_reward, key=lambda x: x[1])[:n]


    def beam_search_core(self, agent, search_depth, search_width, actions_reward, eval_mode, eval_mode2=None):
        if search_depth == 0:
            return

        for action, action_q in self.evaluator.get_actions_q(agent, eval_mode=eval_mode)[:search_width]:
            agent_copy = agent.copy()
            agent_copy.apply_action(action)
            self.beam_search_core(agent_copy, search_depth - 1, search_width, actions_reward, eval_mode)
            
            if eval_mode2 != None:
                action_q = self.evaluator.eval_gflops(agent_copy, eval_mode2)

            actions_reward.append([agent.actions, action_q]) 

class PolicyCostSearcher:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.beam_search = BeamSearcher(evaluator=evaluator)
          
    def search(self, agent, num_steps, search_width, n=5): # actions, reward
        actions_rewards_cost = []
        actions_rewards_policy = self.beam_search.search_n(agent=agent, num_steps=num_steps, search_width=search_width, n=n, eval_mode='policy', eval_mode2='cost')
        print(actions_rewards_policy)
        for actions, _ in actions_rewards_policy:
            agent_copy = agent.copy()
            for action in actions: agent_copy.apply_action(action)
            actions_rewards = self.beam_search.search_n(agent=agent, num_steps=num_steps, search_width=search_width, n=1000, eval_mode='loop_nest')
            actions_rewards_cost.append(actions_rewards)
            
        return max(actions_rewards_cost, key=lambda x: x[1]) 


class GreedySearcher:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.beam_search = BeamSearcher(evaluator=evaluator)

    def search_n(self, agent, num_steps, lookahead, search_width, eval_mode, n): # actions, reward
        actions_reward_pairs = []
        for i in range(n):
            actions, reward = self.search(
                agent=agent,
                num_steps=num_steps,
                lookahead=lookahead,
                search_width=search_width,
                eval_mode=eval_mode,
            )
            actions_reward_pairs.append([actions, reward])

        return max(actions_reward_pairs, key=lambda x: x[1]) 

    def search(self, agent, num_steps, lookahead, search_width, eval_mode): # actions, reward
        agent_copy = agent.copy()
        agent_copy.clear_actions()
        new_reward = 0

        for i in range(num_steps):
            best_actions, new_reward = self.beam_search.search(agent=agent_copy, num_steps=lookahead, search_width=search_width, eval_mode=eval_mode)
            agent_copy.apply_action(best_actions[0])

        return agent_copy.actions, new_reward

    