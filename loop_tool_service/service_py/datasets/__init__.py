from compiler_gym.datasets import Benchmark
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.service.proto import Benchmark as BenchmarkProto, BenchmarkDynamicConfig
from compiler_gym.service.proto import File
from typing import Union


def benchmark_from_file_contents(uri: Union[str, BenchmarkUri], data: bytes,
                                 dynamic_config: BenchmarkDynamicConfig):
    """Construct a benchmark from raw data.

    :param uri: The URI of the benchmark.

    :param data: An array of bytes that will be passed to the compiler
        service.

    :param dynamic_config: BenchmarkDynamicConfig that specifies build, pre_run, run,
        post_run commands.
    """
    return Benchmark(proto=BenchmarkProto(uri=str(uri), program=File(contents=data),
                                          dynamic_config=dynamic_config))
