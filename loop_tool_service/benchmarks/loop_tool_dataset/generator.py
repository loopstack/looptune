import loop_tool as lt

import sys
# import loop_tool_service_py.ui as ui
import numpy as np
import pickle
import pdb


def mm(A, B):
    s = lt.SymbolGenerator()
    C = A.to(s.m, s.k) * B.to(s.k, s.n)
    return C.sum(s.k)

def conv(X, W):
        s = lt.SymbolGenerator()
        X = X.pad(X.symbolic_shape[1], 1)
        return (X[s.B, s.No + s.K] * W.to(s.B, s.K)).sum(s.K)

# ********************************** mm.txt ********************************** 
def gen_mm():
    m, n, k = 128, 128, 128  # 8, 16, 128
    A = lt.Tensor(m, k).set(np.random.randn(m, k))
    B = lt.Tensor(k, n).set(np.random.randn(k, n))

    s = lt.SymbolGenerator()
    # C = mm(A, B).to(s.m, s.n).sum(s.m)  # * A.to(s.m, s.k)
    C = mm(A, B)

    return C
# ********************************** conv.txt **********************************
def gen_conv():
    X = lt.Tensor(256, 128).set(np.random.randn(256, 128))
    W = lt.Tensor(256, 3).set(np.random.randn(256, 3))

    C = conv(X, W)
    return C
# ********************************** muladd.txt **********************************
def gen_muladd():
    A = lt.Tensor(128,128)
    B = lt.Tensor(128,128)
    m, n, k = lt.symbols("m n k")


    C = (A.to(m, k) * B.to(k, n)).sum(k)

    return C
# ********************************** simple.txt **********************************
def gen_simple():
    m, n, k = 128, 128, 128
    A = lt.Tensor(m, k).set(np.random.randn(m, k))
    B = lt.Tensor(k, n).set(np.random.randn(k, n))

    s = lt.SymbolGenerator()
    C = mm(A, B)

    return C
# ********************************** mm128.txt **********************************
def gen_mm128():
    m, n, k = 128, 128, 128
    A = lt.Tensor(m, k).set(np.random.randn(m, k))
    B = lt.Tensor(k, n).set(np.random.randn(k, n))

    s = lt.SymbolGenerator()
    C = mm(A, B)

    loop_tree = C.loop_tree.split(0, 16)\
                           .split(2, 16)\
                           .split(4, 16)

    C.set(loop_tree)

    return C
# ********************************** mm256.txt **********************************
def gen_mm256():
    m, n, k = 256, 256, 256
    A = lt.Tensor(m, k).set(np.random.randn(m, k))
    B = lt.Tensor(k, n).set(np.random.randn(k, n))

    s = lt.SymbolGenerator()
    C = mm(A, B)

    # TODO: BWasti reproducer
    # loop_tree = C.loop_tree.split(0, 4)\
    #               .swap_loops(1, 2)\
    #               .swap_loops(2, 3)\
    #               .swap_loops(2, 1)\
    #               .split(1, 16)\
    #               .swap_loops(2, 3)\
    #               .swap_loops(3, 4)\
    #               .copy_input(5, 0)\
    #               .try_swap(5, 4)\
    #               .split(5, 4)\
    #               .copy_input(7, 1)\
    #               .decrease_reuse(7)\
    #               .decrease_reuse(7)\
    #               .decrease_reuse(7)\
    #               .split(14, 4)

    loop_tree = C.loop_tree.split(0, 16)\
                .swap_loops(1, 2)\
                .swap_loops(2, 3)\
                .swap_loops(2, 1)\
                .split(1, 16)\
                .swap_loops(2, 3)\
                .swap_loops(3, 4)\
                .split(8, 16)
    # Good schedule
    loop_tree = C.loop_tree\
                .split(0, 8)\
                .split(0, 4)\
                .split(3, 8)\
                .split(5, 8)\
                .split(5, 8)\
                .swap_loops(2, 3)\
                .swap_loops(3, 4)\
                .swap_loops(4, 5)\
                .swap_loops(5, 6)\
                .swap_loops(3, 4)\
                .swap_loops(4, 5)\
                .swap_loops(2, 3)\
                .swap_loops(3, 4)
    # Bad Schedule
    loop_tree = C.loop_tree\
                .split(0, 8)\
                .split(0, 4)\
                .split(3, 8)\
                .split(5, 8)\
                .split(5, 8)\
                .swap_loops(2, 3)\
                .swap_loops(3, 4)\
                .swap_loops(4, 5)\
                .swap_loops(5, 6)\
                .swap_loops(3, 4)\
                .swap_loops(4, 5)\
                .swap_loops(2, 3)\
                .swap_loops(3, 4)\
                .swap_loops(5, 6)\
                .swap_loops(6, 7)

    C.set(loop_tree)

    return C
# # ********************************** mm512.txt **********************************
def gen_mm512():
    m, n, k = 512, 512, 512
    A = lt.Tensor(m, k).set(np.random.randn(m, k))
    B = lt.Tensor(k, n).set(np.random.randn(k, n))

    s = lt.SymbolGenerator()
    C = mm(A, B)

    return C





def main():
    
    print(sys.argv)
    if len(sys.argv) not in [2, 3]:
        print('Format: generator.py file [tune=False]')
        return 

    path_to_file = sys.argv[1]
    file_name = path_to_file.split('/')[-1]
    tune = True if len(sys.argv) == 3 else False

    if file_name == 'conv.txt':
        C = gen_conv()
    elif file_name == 'mm.txt':
        C = gen_mm()
    elif file_name == 'mm128.txt':
        C = gen_mm128()
    elif file_name == 'mm256.txt':
        C = gen_mm256()
    elif file_name == 'mm512.txt':
        C = gen_mm512()
    elif file_name == 'muladd.txt':
        C = gen_muladd()        
    elif file_name == 'simple.txt':
        C = gen_simple()
    else:
        print('File not found :/')

    agent = lt.LoopTreeAgent(lt.LoopTree(C.ir))
    breakpoint()
    agent.apply_action("unroll")
    breakpoint()

    if tune:
        with lt.Backend("loop_nest"):
            C = lt.ui(C, "/tmp/woo.c")

    with open(file_name, "w") as f:
        f.write(C.ir.serialize())
   

if __name__ == '__main__':
    main()
