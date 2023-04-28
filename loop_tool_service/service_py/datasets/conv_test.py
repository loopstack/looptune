from compiler_gym.datasets import Benchmark, BenchmarkUri, Dataset
from compiler_gym.util.runfiles_path import site_data_path
from typing import Iterable
from pathlib import Path
import os

from loop_tool_service.paths import BENCHMARKS_PATH

from . import benchmark_from_file_contents
from compiler_gym.service.proto import BenchmarkDynamicConfig, Command


bench_name = os.path.basename(__file__).strip('.py')

BENCHMARKS_PATH = BENCHMARKS_PATH/bench_name


class Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        dataset_name = f"benchmark://{bench_name}-v0"

        super().__init__(
            name=dataset_name,
            license="MIT",
            description="LoopTool test dataset",
            site_data_base=site_data_path("example_dataset"),
        )

        benchmark_config = BenchmarkDynamicConfig(
                    build_cmd=Command(
                        argument=["$CC", "$IN"],
                        timeout_seconds=60000,
                        outfile=["a.out"],
                    ),
                    run_cmd=Command(
                        argument=["./a.out"],
                        timeout_seconds=30000,
                        infile=["a.out"],
                    )
                )

        self._benchmarks = {}
        benchmark_prefix = f"benchmark://{bench_name}-v0"

        example_files = os.listdir(BENCHMARKS_PATH)
        for i, example_filename in enumerate(example_files):
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
