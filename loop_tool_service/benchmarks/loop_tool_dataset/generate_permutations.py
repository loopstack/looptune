import loop_tool as lt
import argparse

from itertools import combinations, permutations
from loop_tool_service.paths import LOOP_TOOL_ROOT
import os
import shutil




# def mm(A, B):
#     s = lt.SymbolGenerator()
#     C = A.to(s.m, s.k) * B.to(s.k, s.n)
#     return C.sum(s.k)

# def gen_mm():
#     m, n, k = 128, 128, 128  # 8, 16, 128
#     A = lt.Tensor(m, k).set(np.random.randn(m, k))
#     B = lt.Tensor(k, n).set(np.random.randn(k, n))

#     s = lt.SymbolGenerator()
#     C = mm(A, B)

#     loop_tree = C.loop_tree.split(0, 16)\
#                              .split(2, 16)\
#                              .split(4, 16)
#     C.set(loop_tree)
#     return C


def swap_loops(loop_tree, l1, l2):
    if loop_tree.loop(l1).var == loop_tree.loop(l2).var:
        for l in loop_tree.loops:
            if loop_tree.loop(l).var != loop_tree.loop(l1).var:
                return loop_tree.swap_loops(l, l1)\
                                .swap_loops(l1, l2)\
                                .swap_loops(l, l2)
    else:
        return loop_tree.swap_loops(l1, l2)


def order_loop_tree(loop_tree, order):
    i = 0
    order_tmp = list(order)

    while(i < len(order_tmp)):
        if i == order_tmp[i]:
            i += 1
        else:
            # breakpoint()
            loop_tree = swap_loops(loop_tree, l1=i, l2=order[i])
            order_tmp[order_tmp[i]], order_tmp[i] = order_tmp[i], order_tmp[order_tmp[i]]

    return loop_tree

def generate_permutations(loop_tree):
    loop_ids = [ x for i, x in enumerate(loop_tree.loops) if x == i ]
    loops = []
    orders = []
    for comb in combinations(loop_ids, len(loop_ids)):
        for perm in permutations(comb):
            loops.append(order_loop_tree(loop_tree, perm))
            orders.append(perm)
    return loops, orders



dataset_path = LOOP_TOOL_ROOT/"loop_tool_service/benchmarks/loop_tool_dataset/mmo_perm"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bench", type=str, default="mm.txt", help="Benchmark to generate"
)
parser.add_argument(
    "--out", type=str, default=dataset_path, help="Path to folder with permutations to generate"
)

args = parser.parse_args()


def main():

    with open(args.bench, 'r') as f: ir = lt.deserialize(f.read())
    loop_tree = lt.LoopTree(ir)

    if os.path.exists(args.out): shutil.rmtree(args.out)
    os.makedirs(args.out)
    
    loops, orders = generate_permutations(loop_tree)
    for loop_tree, order in zip(loops, orders):
        print(loop_tree)
        with open(f'{args.out}/{"".join([ str(x) for x in order])}.txt', "w") as f:
            f.write(ir.serialize())
   

if __name__ == '__main__':
    main()
