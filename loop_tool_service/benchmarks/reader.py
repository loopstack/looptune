'''
python reader.py --bench=mm64_256_16_range/mm112_208_256 --actions="['split_4', 'down', 'swap_down', 'down', 'split_16', 'swap_up', 'swap_up', 'swap_up', 'split_8', 'swap_down']"
'''

import argparse
import loop_tool as lt

import numpy as np
import pandas as pd
import json


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bench", type=str, help="Benchmark to read", required=True
)
parser.add_argument(
    "--actions", type=str, default='[]', help="Actions to apply",
)

args = parser.parse_args()




def mm(A, B):
    s = lt.SymbolGenerator()
    C = A.to(s.m, s.k) * B.to(s.k, s.n)
    return C.sum(s.k)

def gen_mm():
    m, n, k = 128, 128, 128  # 8, 16, 128
    A = lt.Tensor(m, k).set(np.random.randn(m, k))
    B = lt.Tensor(k, n).set(np.random.randn(k, n))

    s = lt.SymbolGenerator()
    C = mm(A, B)

    return C

if args.bench.endswith('.pkl'):
    df = pd.read_pickle(args.bench) 
    df.head()
    breakpoint()
else:
    with open(args.bench, 'r') as f: ir = lt.deserialize(f.read())
    C = gen_mm()

    agent = lt.LoopTreeAgent(lt.LoopTree(ir)).merge_all()
    with lt.Backend("loop_nest"):
        print(agent.eval("FLOPS") / 1e9)
        print(agent)

    for action in json.loads(args.actions.replace("'", '"')):
        if action in agent.get_available_actions():
            agent.apply_action(action)
            with lt.Backend("loop_nest"):
                print(agent.eval("FLOPS") / 1e9)
                print(agent)

    C = C.set(agent.lt.ir)


    with lt.Backend("loop_nest"):
        breakpoint()
        C = lt.ui(C, "/tmp/woo.c")

    breakpoint()
