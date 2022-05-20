import loop_tool as lt

# import loop_tool_service_py.ui as ui
import numpy as np
import pickle
import pdb

def mm(A, B):
    s = lt.SymbolGenerator()
    C = A.to(s.m, s.k) * B.to(s.k, s.n)
    return C.sum(s.k)


m, n, k = 128, 128, 128  # 8, 16, 128
A = lt.Tensor(m, k).set(np.random.randn(m, k))
B = lt.Tensor(k, n).set(np.random.randn(k, n))

s = lt.SymbolGenerator()
C = mm(A, B).to(s.m, s.n).sum(s.m)  # * A.to(s.m, s.k)


def conv(X, W):
    s = lt.SymbolGenerator()
    X = X.pad(X.symbolic_shape[1], 1)
    return (X[s.B, s.No + s.K] * W.to(s.B, s.K)).sum(s.K)


X = lt.Tensor(256, 128).set(np.random.randn(256, 128))
W = lt.Tensor(256, 3).set(np.random.randn(256, 3))

C = conv(X, W)
with open("data/conv.txt", "w") as f:
    f.write(C.ir.serialize())


A = lt.Tensor(m, k).set(np.random.randn(m, k))
B = lt.Tensor(m, k).set(np.random.randn(m, k))
C = mm(A, B)

with open("data/mm.txt", "w") as f:
    f.write(C.ir.serialize())

pdb.set_trace()
lt.ui(C, "/tmp/woo.c")
