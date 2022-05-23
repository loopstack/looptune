import loop_tool as lt
import pdb
pdb.set_trace()
A = lt.Tensor(128,128)
B = lt.Tensor(128,128)
m, n, k = lt.symbols("m n k")


C = (A.to(m, k) * B.to(k, n)).sum(k)

with open("data/muladd.txt", "w") as f:
    f.write(C.ir.serialize())