import tvm
import tvm.testing
from tvm import te
import numpy
import timeit
import pandas as pd


def tvm_mm(benchmark_name, search_cmd, debug=False):
    columns = search_cmd.split(',')
    M, K, N = benchmark_name.lstrip('mm').split('_')

    result = {x:0 for x in columns}

    # # The size of the matrix
    # # (M, K) x (K, N)
    # # You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
    # M = 1024
    # K = 1024
    # N = 1024

    # The default tensor type in tvm
    dtype = "float32"

    # using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
    # To get the best performance, please change the following line
    # to llvm -mcpu=core-avx2, or specific type of CPU you use
    target = "llvm"
    dev = tvm.device(target, 0)

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    if 'tvm_numpy' in columns:
        np_repeat = 100
        np_runing_time = timeit.timeit(
            setup="import numpy\n"
            "M = " + str(M) + "\n"
            "K = " + str(K) + "\n"
            "N = " + str(N) + "\n"
            'dtype = "float32"\n'
            "a = numpy.random.rand(M, K).astype(dtype)\n"
            "b = numpy.random.rand(K, N).astype(dtype)\n",
            stmt="answer = numpy.dot(a, b)",
            number=np_repeat,
        )
        print("Numpy running time: %f" % (np_runing_time / np_repeat))
        result['numpy'] = np_runing_time / np_repeat

    answer = numpy.dot(a.numpy(), b.numpy())

    # Algorithm
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

    # Default schedule
    s = te.create_schedule(C.op)
    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    tvm_base = evaluator(a, b, c).mean
    print(f"Baseline: {tvm_base}")
    result['tvm_base'] = tvm_base

    if debug:
        print(tvm.lower(s, [A, B, C], simple_mode=True))

    # Blocking __________________________________________________________________________
    if 'tvm_blocking' in columns:
        bn = 32
        kfactor = 4
        s = te.create_schedule(C.op)

        # Blocking by loop tiling
        mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
        (kaxis,) = s[C].op.reduce_axis
        ko, ki = s[C].split(kaxis, factor=kfactor)

        # Hoist reduction domain outside the blocking loop
        s[C].reorder(mo, no, ko, ki, mi, ni)

        func = tvm.build(s, [A, B, C], target=target, name="mmult")
        assert func

        c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
        func(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

        # By simply tiling the loop 32x32, and hoisting ko, ki outside the blocking loops,
        # we can see big speedup compared with the baseline.
        evaluator = func.time_evaluator(func.entry_name, dev, number=10)
        tvm_blocking = evaluator(a, b, c).mean
        print(f"tvm_blocking: {tvm_blocking}")
        result['tvm_blocking'] = tvm_blocking

        if debug:
            print(tvm.lower(s, [A, B, C], simple_mode=True))

    # Vectorization ___________________________________________________________
    if 'tvm_vectorization' in columns:
        s = te.create_schedule(C.op)
        mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
        (kaxis,) = s[C].op.reduce_axis
        ko, ki = s[C].split(kaxis, factor=kfactor)

        s[C].reorder(mo, no, ko, ki, mi, ni)

        # Vectorization
        s[C].vectorize(ni)

        func = tvm.build(s, [A, B, C], target=target, name="mmult")
        assert func

        c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
        func(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

        evaluator = func.time_evaluator(func.entry_name, dev, number=10)
        tvm_vectorization = evaluator(a, b, c).mean
        print(f"tvm_vectorization: {tvm_vectorization}")
        result['tvm_vectorization'] = tvm_vectorization

        if debug:
            print(tvm.lower(s, [A, B, C], simple_mode=True))
        

    # LoopPermutation _________________________________________________________
    if 'tvm_permutation' in columns:
        s = te.create_schedule(C.op)
        mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
        (kaxis,) = s[C].op.reduce_axis
        ko, ki = s[C].split(kaxis, factor=kfactor)

        # re-ordering
        s[C].reorder(mo, no, ko, mi, ki, ni)
        s[C].vectorize(ni)

        func = tvm.build(s, [A, B, C], target=target, name="mmult")
        assert func

        c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
        func(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

        evaluator = func.time_evaluator(func.entry_name, dev, number=10)
        tvm_permutation = evaluator(a, b, c).mean
        print(f"tvm_permutation: {tvm_permutation}")
        result['tvm_permutation'] = tvm_permutation

        if debug:
            print(tvm.lower(s, [A, B, C], simple_mode=True))


    # Array Packing _________________________________________________________
    if 'tvm_packing' in columns:
        # We have to re-write the algorithm slightly.
        packedB = te.compute(
            (N / bn, K, bn), lambda bigN, k, littleN: B[k, bigN * bn + littleN], name="packedB"
        )
        C = te.compute(
            (M, N),
            lambda m, n: te.sum(A[m, k] * packedB[n // bn, k, tvm.tir.indexmod(n, bn)], axis=k),
            name="C",
        )

        s = te.create_schedule(C.op)

        mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
        (kaxis,) = s[C].op.reduce_axis
        ko, ki = s[C].split(kaxis, factor=kfactor)

        s[C].reorder(mo, no, ko, mi, ki, ni)
        s[C].vectorize(ni)

        bigN, _, littleN = s[packedB].op.axis
        s[packedB].vectorize(littleN)
        s[packedB].parallel(bigN)

        func = tvm.build(s, [A, B, C], target=target, name="mmult")
        assert func

        c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
        func(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

        evaluator = func.time_evaluator(func.entry_name, dev, number=10)
        tvm_packing = evaluator(a, b, c).mean
        print(f"tvm_packing: {tvm_packing}")
        result['tvm_packing'] = tvm_packing

        if debug:
            print(tvm.lower(s, [A, B, C], simple_mode=True))
        


    # Write cache for blocks_________________________________________________________
    if 'tvm_cache_blocking' in columns:
        s = te.create_schedule(C.op)

        # Allocate write cache
        CC = s.cache_write(C, "global")

        mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

        # Write cache is computed at no
        s[CC].compute_at(s[C], no)

        # New inner axes
        mc, nc = s[CC].op.axis

        (kaxis,) = s[CC].op.reduce_axis
        ko, ki = s[CC].split(kaxis, factor=kfactor)
        s[CC].reorder(ko, mc, ki, nc)
        s[CC].vectorize(nc)

        # TODO: Add separate optimization step to discuss loop unrolling
        # unrolling is a loop optimization strategy which can reduce branch
        # prediction failures and increases the chance of concurrent execution
        # unroll kfactor loops
        s[CC].unroll(ki)

        bigN, _, littleN = s[packedB].op.axis
        s[packedB].vectorize(littleN)
        s[packedB].parallel(bigN)

        func = tvm.build(s, [A, B, C], target=target, name="mmult")
        assert func

        c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
        func(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

        evaluator = func.time_evaluator(func.entry_name, dev, number=10)
        tvm_cache_blocking = evaluator(a, b, c).mean
        print(f"tvm_cache_blocking: {tvm_cache_blocking}")
        result['tvm_cache_blocking'] = tvm_cache_blocking

        if debug:
            print(tvm.lower(s, [A, B, C], simple_mode=True))
        


    # Multi-core_________________________________________________________
    if 'tvm_parallel' in columns:
        s = te.create_schedule(C.op)

        CC = s.cache_write(C, "global")

        mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

        s[CC].compute_at(s[C], no)

        mc, nc = s[CC].op.axis

        (kaxis,) = s[CC].op.reduce_axis
        ko, ki = s[CC].split(kaxis, factor=kfactor)
        s[CC].reorder(ko, mc, ki, nc)
        s[CC].vectorize(nc)
        s[CC].unroll(ki)

        # parallel
        s[C].parallel(mo)

        bigN, _, littleN = s[packedB].op.axis
        s[packedB].vectorize(littleN)
        s[packedB].parallel(bigN)

        func = tvm.build(s, [A, B, C], target=target, name="mmult")
        assert func

        c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
        func(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

        evaluator = func.time_evaluator(func.entry_name, dev, number=50)
        tvm_parallel = evaluator(a, b, c).mean
        print(f"tvm_parallel: {tvm_parallel}")
        result['tvm_parallel'] = tvm_parallel

        if debug:
            print(tvm.lower(s, [A, B, C], simple_mode=True))


    return [], result, []

