"""
Reproduce results from wandb:
$ python compare_to_tvm.py --wandb_url=dejang/loop_tool_agent_split/61e41_00000
"""


import argparse
import ray
from loop_tool_service.models.rllib.rllib_agent import RLlibAgent
import tvm
import tvm.testing
from tvm import te
from tvm import autotvm

import numpy
import timeit
import pandas as pd
from matplotlib import pyplot as plt
import wandb
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wandb_url",  type=str, nargs='?', default='dejang/loop_tool_agent_split/61e41_00000', help="Wandb uri to load policy network."
)
parser.add_argument(
    "--size",  type=int, nargs='?', default=20, help="Number of benchmarks to evaluate."
)


@autotvm.template("tutorial/matmul")
def matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


def mm_autotvm(M, K, N):
    task = autotvm.task.create("tutorial/matmul", args=(M, K, N, "float32"), target="llvm")
    print(task.config_space)
    measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))
    # tuner = autotvm.tuner.RandomTuner(task)
    tuner = autotvm.tuner.XGBTuner(task)
    # :any:tvm.autotvm.tuner.RandomTuner: Enumerate the space in a random order
    # :any:tvm.autotvm.tuner.GridSearchTuner: Enumerate the space in a grid search order
    # :any:tvm.autotvm.tuner.GATuner: Using genetic algorithm to search through the space
    # :any:tvm.autotvm.tuner.XGBTuner
    tuner.tune(
        n_trial=10,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("matmul.log")],
    )

    # apply history best from log file
    with autotvm.apply_history_best("matmul.log"):
        with tvm.target.Target("llvm"):
            s, arg_bufs = matmul(M, K, N, "float32")
            func = tvm.build(s, arg_bufs)

    
    target = 'llvm -mcpu=core-avx2'
    dev = tvm.device(target, 0)
    dtype = "float32"
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)

    func(a, b, c)

    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    tvm_base = evaluator(a, b, c).mean
    return {'auto_tvm': tvm_base} 




def mm_tvm(M, K, N, columns, debug=False):
    result = {x:0 for x in columns}

    # # The size of the matrix
    # # (M, K) x (K, N)

    # The default tensor type in tvm
    dtype = "float32"

    # using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
    # To get the best performance, please change the following line
    # to llvm -mcpu=core-avx2, or specific type of CPU you use
    # target = "llvm"
    target = 'llvm -mcpu=core-avx2'
    dev = tvm.device(target, 0)

    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)
    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)

    if 'numpy' in columns:
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

    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    tvm_base = evaluator(a, b, c).mean
    print(f"Baseline: {tvm_base}")
    result['tvm_base'] = tvm_base

    if debug:
        print(tvm.lower(s, [A, B, C], simple_mode=True))

    # Blocking __________________________________________________________________________
    if '+tvm_blocking' in columns:
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
        print(f"+tvm_blocking: {tvm_blocking}")
        result['+tvm_blocking'] = tvm_blocking

        if debug:
            print(tvm.lower(s, [A, B, C], simple_mode=True))

    # LoopPermutation _________________________________________________________
    if '+tvm_permutation' in columns:
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
        print(f"+tvm_permutation: {tvm_permutation}")
        result['+tvm_permutation'] = tvm_permutation

        if debug:
            print(tvm.lower(s, [A, B, C], simple_mode=True))


    # Vectorization ___________________________________________________________
    if '+tvm_vectorization' in columns:
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
        print(f"+tvm_vectorization: {tvm_vectorization}")
        result['+tvm_vectorization'] = tvm_vectorization

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
    if '+tvm_parallel' in columns:
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
        print(f"+tvm_parallel: {tvm_parallel}")
        result['+tvm_parallel'] = tvm_parallel

        if debug:
            print(tvm.lower(s, [A, B, C], simple_mode=True))

    return result

def get_benchmarks(wandb_url):
    api = wandb.Api()
    wandb_run = api.run(wandb_url)
    return wandb_run.summary['test_benchmarks']


def get_matrix_sizes(benchmarks):
    matrix_sizes = []
    for benchmark_url in benchmarks:
        m, k, n = benchmark_url.split('/')[-1].lstrip('mm').split('_')
        matrix_sizes.append([benchmark_url, int(m), int(k), int(n)])
    return matrix_sizes


def init_looptune(wandb_url):
    agent = RLlibAgent(
        trainer='apex_dqn.ApexDQN', #'dqn.ApexTrainer', 
        # dataset='mm64_256_16_range', 
        size=10000000, 
        eval_size=2,
        network='TorchCustomModel', 
        sweep_count=0, 
        eval_time=10
    )

    agent.load_model(wandb_url)
    
    agent.config["explore"] = False
    policy_model = agent.trainer(
        env="compiler_gym",
        config=agent.config
    )

    policy_model.restore(str(agent.checkpoint_path))
    agent.evaluator.reward = 'runtime'
    agent.evaluator.set_policy_agent(policy_model)

    return agent

def eval_looptune(agent, benchmark):
    results_gflops, results_time, results_actions = agent.evaluator.evaluate_single_benchmark(agent.env, benchmark, searches=['looptune'])

    return {'looptune':results_gflops['looptune'][-1]}


def plot_bars(df):    
    breakpoint()
    avg = df.mean(axis=0).to_dict()
    df_mean = df.apply(['mean'])
    df_mean['benchmark'] = 'average'
    df = pd.concat([df, df_mean], ignore_index=True)
    
    figsize = ((len(df) + 1) // 2, 5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = df.plot(x='benchmark', y=df.columns[1:], kind='bar', figsize=figsize, width=0.8, align='edge', ax=ax)
    plt.minorticks_on()
    plt.grid(which='both', axis='y')
    ax.set_ylabel('execution time [s]')
    ax.set_yscale('log')
    ax.legend(title='Searches',loc='center left', bbox_to_anchor=(1, 0.5))


    plt.gca().yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10, subs='all', numticks=100))

    plt.suptitle(f'Benchmarks evaluation', fontsize=16)
    fig.autofmt_xdate()

    fig.savefig(f'compare_tvm_bars.png', bbox_inches = 'tight')


def main():
    args = parser.parse_args()

    ray.init(local_mode=False, ignore_reinit_error=True)

    agent = init_looptune(args.wandb_url)
    benchmarks = get_benchmarks(args.wandb_url)[:args.size]
    matrix_sizes_list = get_matrix_sizes(benchmarks)

    columns = ['numpy', 'tvm_base', '+tvm_blocking', '+tvm_permutation', '+tvm_vectorization']
    df = pd.DataFrame(columns=columns)

    for benchmark_url, m, k, v in matrix_sizes_list:
        results = {'benchmark': benchmark_url.split('/')[-1]}
        results.update(mm_tvm(m, k, v, columns))
        results.update(mm_autotvm(m, k, v))
        results.update(eval_looptune(agent, benchmark_url))
        df = df.append(results, ignore_index=True)

    
    df = df[results.keys()]

    plot_bars(df)
    df.to_csv('compare_tvm_results.csv')

    ray.shutdown()
    print("Return from train!")


if __name__ == '__main__':
    main()