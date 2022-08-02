import loop_tool as lt
import sys
from loop_tool_service.paths import BENCHMARKS_PATH

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Format: python handtune_benchmark.py benchmark_uri')
        exit()


    target_uri = sys.argv[1]

    file = list(BENCHMARKS_PATH.glob(f'**/{target_uri}.txt'))[0]
    print(file)



    # irs =    '\
    # v:m_786759241\n\
    # v:k_786759243\n\
    # v:n_786759242\n\
    # n:2::0,1,:::0:::::\n\
    # n:2::1,2,:::0:::::\n\
    # n:7:0,1,:0,1,2,:::0:0;10;7,0;12;0,1;3;21,1;36;0,2;7;15,2;16;0,::,,,,,,::\n\
    # n:5:2,:0,2,:::0:0;10;7,0;12;0,1;3;21,1;36;0,2;7;15,2;16;0,::,,,,,,::\n\
    # n:1:3,:0,2,:::0:0;10;7,0;12;0,2;127;0,::,,,::\n\
    # i:0,1,\n\
    # o:4,\n\
    # '
    # ir = lt.deserialize(irs)


    with open(file, 'r') as f:
        ir = lt.deserialize(f.read())
        
    tree = lt.LoopTree(ir)
    print(tree)

    with lt.Backend("loop_nest"): print(tree.FLOPS())

    C = lt.Tensor()
    C.set(tree)
    C = lt.ui(C, "/tmp/woo.c")