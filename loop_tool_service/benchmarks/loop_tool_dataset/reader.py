import loop_tool as lt

import numpy as np
import pdb

C = lt.Tensor()

with open("simple_loops/mm.txt", "r") as f:
    tmp = lt.deserialize(f.read())

C.set(tmp)
pdb.set_trace()

print(C.code)
