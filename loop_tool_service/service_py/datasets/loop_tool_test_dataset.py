from compiler_gym.datasets import Benchmark, BenchmarkUri, Dataset
from compiler_gym.util.runfiles_path import site_data_path
from compiler_gym.third_party import llvm
from typing import Iterable
import subprocess
from pathlib import Path
import pdb
import sys
import os

from loop_tool_service.paths import BENCHMARKS_PATH

from compiler_gym.envs.llvm.llvm_benchmark import get_system_library_flags
from . import benchmark_from_file_contents
from compiler_gym.service.proto import BenchmarkDynamicConfig, Command


BENCHMARKS_PATH = BENCHMARKS_PATH/"loop_tool_dataset/small"


class Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://loop_tool_test-v0",
            license="MIT",
            description="LoopTool test dataset",
            site_data_base=site_data_path("example_dataset"),
        )

        benchmark_config = BenchmarkDynamicConfig(
                    build_cmd=Command(
                        argument=["$CC", "$IN"],
                        timeout_seconds=60,
                        outfile=["a.out"],
                    ),
                    run_cmd=Command(
                        argument=["./a.out"],
                        timeout_seconds=3000,
                        infile=["a.out"],
                    )
                )

        self._benchmarks = {}
        benchmark_prefix = "benchmark://loop_tool_test-v0"

        example_files = os.listdir(BENCHMARKS_PATH)
        for i, example_filename in enumerate(example_files):
            if i == 100: break
            example_uri = benchmark_prefix + '/' + example_filename.rstrip('.txt')
            self._benchmarks[example_uri] = \
                benchmark_from_file_contents(
                    example_uri,
                    self.preprocess(BENCHMARKS_PATH / example_filename),
                    benchmark_config
                ) 
            

                

    @property
    def size(self) -> int:
        return len(self._benchmarks)

    def __len__(self) -> int:
        return self.size
        
    @staticmethod
    def preprocess(src: Path) -> str:
        """Front a C source through the compiler frontend."""
        code_dump = None
        with open(src, "r") as f:
            code_dump = f.read()
        return str.encode(code_dump)

    def benchmark_uris(self) -> Iterable[str]:
        yield from self._benchmarks.keys()

    def benchmark(self, uri: str) -> Benchmark:
        if uri in self._benchmarks:
            return self._benchmarks[uri]
        else:
            raise LookupError("Unknown program name")

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        # TODO: IMPORTANT
        return self.benchmark(str(uri))
