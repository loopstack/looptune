import argparse
import loop_tool as lt

import numpy as np
import pdb


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bench", type=str, help="Benchmark to generate", required=True
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


with open(args.bench, 'r') as f: ir = lt.deserialize(f.read())
C = gen_mm()
C = C.set(ir)

with lt.Backend("loop_nest"):
    C = lt.ui(C, "/tmp/woo.c")
