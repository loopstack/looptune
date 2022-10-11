'''
This file generates benchmark by leting user define loop order for matrix multiplicaiton
or convolution, and making all permutations of it enumerated with 01234...txt and saving 
it under args.out directory

Example:

python generate_permutations.py --kind=mm --size=128 --out=path-to-dir [--permute]
'''

import argparse
import sys
import loop_tool as lt
import numpy as np

from itertools import combinations, permutations
from loop_tool_service.paths import LOOP_TOOL_ROOT
import os
import shutil



parser = argparse.ArgumentParser(description='Generate dataset of permutation for the constructed looptree.')

parser.add_argument(
    '--kind', choices=['mm', 'conv'], help='Kind of computation.', required=True
)
parser.add_argument(
    "--dimA",  type=str, help="Dimensions of the tensor A.", required=True
)
parser.add_argument(
    "--dimB",  type=str, help="Dimensions of the tensor B.", required=True
)
parser.add_argument(
    "--out", type=str, help="Path to folder with permutations to generate.", required=True
)
parser.add_argument(
    "--permute",  type=bool, nargs='?', const=True, default=False, help="Generate all permutations of loops."
)


def mm(A, B):
    s = lt.SymbolGenerator()
    C = A.to(s.m, s.k) * B.to(s.k, s.n)
    return C.sum(s.k)

def conv(X, W):
    s = lt.SymbolGenerator()
    X = X.pad(X.symbolic_shape[1], 1)
    return (X[s.B, s.No + s.K] * W.to(s.B, s.K)).sum(s.K)


# ********************************** mm.txt ********************************** 
def gen_mm(dimA, dimB):
    assert(dimA[1] == dimB[0])
    m, n, k = dimA[0], dimA[1], dimB[1]  # 8, 16, 128
    A = lt.Tensor(m, k).set(np.random.randn(m, k))
    B = lt.Tensor(k, n).set(np.random.randn(k, n))

    s = lt.SymbolGenerator()
    C = mm(A, B)
    return C
# ********************************** conv.txt **********************************
def gen_conv(dimA, dimB):
    n, c, h, w = lt.symbols('n c h w')
    X = lt.Tensor(*dimA).to(n, c, h, w) # 64, 8, 24, 24

    m, kh, kw = lt.symbols('m kh kw')
    W = lt.Tensor(*dimB).to(m, c, kh, kw) # 16, 8, 3, 3

    Y = lt.nn.conv(X, W, [h, w], [kh, kw])

    # nhwc output by default, we can fix that
    _, ho, wo, _ = Y.symbolic_shape
    Y = Y.transpose(n, m, ho, wo)

    print(Y.shape)
    print(Y.loop_tree)
    return Y




# ********************************** permutations.txt **********************************

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




def main():

    args = parser.parse_args()
    dimA = [ int(item) for item in args.dimA.split(',') ]
    dimB = [ int(item) for item in args.dimA.split(',') ] 

    if args.kind == "mm":
        C = gen_mm(dimA, dimB)
    elif args.kind == "conv":
        C = gen_conv(dimA, dimB)
    else:
        breakpoint()
        exit()

    with lt.Backend("loop_nest"):
        C = lt.ui(C, "/tmp/woo.c")


    assert(not os.path.exists(args.out)), f"File: {args.out} already exist!"
    
    if args.permute:
        os.makedirs(args.out)
        loops, orders = generate_permutations(C.loop_tree)
        for loop_tree, order in zip(loops, orders):
            print(loop_tree)
            with open(f'{args.out}/{"".join([ str(x) for x in order])}.txt', "w") as f:
                f.write(loop_tree.ir.serialize())
    else:
        with open(args.out, "w") as f:
            f.write(C.ir.serialize())

if __name__ == '__main__':
    main()
