import random
import networkx as nx

class BeamSearcherCore:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def search(self, agent, num_steps, search_width, eval_mode, graph=None): # pair(actions, reward)
        assert (num_steps >= 0), "BeamSearcher num steps must be >= 0"
        if num_steps == 0:
            return [ random.choice(agent.get_available_actions()) ], 0
        else:
            return self.search_n(
                agent=agent, 
                num_steps=num_steps, 
                search_width=search_width,
                n=1,
                eval_mode=eval_mode,
                graph=graph,
            )[0]

    def search_n(self, agent, num_steps, search_width, n, eval_mode, graph=None):
        agent_copy = agent.copy()
        agent_copy.clear_actions()
        actions_reward = []

        self.beam_search_core(
            agent=agent_copy, 
            search_depth=num_steps, 
            search_width=search_width,
            actions_reward = actions_reward,
            eval_mode=eval_mode,
            graph=graph,
        )
        # breakpoint()
        return sorted(actions_reward, key=lambda x: x[1], reverse=True)[:n]


    def beam_search_core(self, agent, search_depth, search_width, actions_reward, eval_mode, cumulative_reward=0, graph=None):
        if graph != None:
            node_key = hash(agent.dump())
            real_flops = self.evaluator.eval_gflops(agent, 'loop_nest')
            graph.add_node(
                node_key,
                label=f'FLOPS = {real_flops:9.4f}\nPRED = {cumulative_reward:9.4f}\n' + agent.dump().replace(':', ';')
            )
            if len(graph.nodes()) == 1:
                graph.nodes[node_key]['color'] = 'red'
                graph.nodes[node_key]['penwidth'] = 5


        if search_depth == 0:
            if graph != None:
                graph.nodes[node_key]['fillcolor'] = 'lightblue1'
                graph.nodes[node_key]['style'] = 'filled'
            return



        for action, action_q in self.evaluator.get_actions_q(agent, eval_mode=eval_mode)[:search_width]:
            agent_copy = agent.copy()
            agent_copy.apply_action(action)

            if graph != None:
                graph.add_edge(hash(agent.dump()), hash(agent_copy.dump()), key=action, label=f'{action}\n{action_q:9.4f}', color='black')

            cumulative_reward += action_q
            self.beam_search_core(agent_copy, search_depth - 1, search_width, actions_reward, eval_mode, cumulative_reward, graph)
            cumulative_reward -= action_q

            if eval_mode == 'policy':
                actions_reward.append([agent_copy.actions, cumulative_reward + action_q]) 
            else:
                actions_reward.append([agent_copy.actions, action_q]) 



class GreedySearcher:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.beam_search = BeamSearcherCore(evaluator=evaluator)

    def search(self, agent, num_steps, lookahead, search_width, eval_mode, debug=False): # actions, reward
        agent_copy = agent.copy()
        agent_copy.clear_actions()
        new_reward = 0
        graph = nx.MultiDiGraph() if debug else None


        for i in range(num_steps):
            best_actions, new_reward = self.beam_search.search(agent=agent_copy, num_steps=lookahead, search_width=search_width, eval_mode=eval_mode, graph=graph)
            if len(best_actions) == 0: breakpoint()
            agent_copy.apply_action(best_actions[0])

        if debug:
            print(nx.nx_pydot.to_pydot(graph))

        return agent_copy.actions, new_reward


class BeamSearcher:
    """ Uses policy beam search first to find n best candidates. For each candidate apply another beam search.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.beam_search = BeamSearcherCore(evaluator=evaluator)

    def search(self, agent, num_steps, search_width, eval_mode, debug=False):
        graph = nx.MultiDiGraph() if debug else None
        actions_rewards = self.beam_search.search_n(agent=agent, num_steps=num_steps, search_width=search_width, n=search_width, eval_mode=eval_mode, graph=graph)
        if debug:        
            print(nx.nx_pydot.to_pydot(graph))
        
        return max(actions_rewards, key=lambda x: x[1]) 


class BeamBeamSearcher:
    """ Uses policy beam search first to find n best candidates. For each candidate apply another beam search.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.beam_search = BeamSearcherCore(evaluator=evaluator)

          
    def search(self, agent, num_steps1, eval_mode1, search_width1, num_steps2, eval_mode2, search_width2, debug=False): # actions, reward
        actions_rewards_cost = []
        graph = nx.MultiDiGraph() if debug else None
        actions_rewards_policy = self.beam_search.search_n(agent=agent, num_steps=num_steps1, search_width=search_width1, n=search_width1, eval_mode=eval_mode1, graph=graph)
        print(actions_rewards_policy)
        for actions, _ in actions_rewards_policy:
            agent_copy = agent.copy()
            for action in actions: agent_copy.apply_action(action)
            actions_rewards = self.beam_search.search(agent=agent, num_steps=num_steps2, search_width=search_width2, eval_mode=eval_mode2, graph=graph)
            actions_rewards_cost.extend([actions_rewards])

        if debug:        
            print(nx.nx_pydot.to_pydot(graph))
        return max(actions_rewards_cost, key=lambda x: x[1]) 


    