from compiler_gym.datasets import Benchmark, BenchmarkUri, Dataset
from compiler_gym.util.runfiles_path import site_data_path
from compiler_gym.third_party import llvm
from typing import Iterable
from pathlib import Path
from loop_tool_service.paths import BENCHMARKS_PATH

from . import benchmark_from_file_contents
from compiler_gym.service.proto import BenchmarkDynamicConfig, Command


BENCHMARKS_PATH = BENCHMARKS_PATH/"loop_tool_dataset/data"


class Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        dataset_name = "benchmark://single_mm-v0"
        super().__init__(
            name=dataset_name,
            license="MIT",
            description="Single matrix multiply 128 dataset",
            site_data_base=site_data_path("example_dataset"),
        )

        benchmark_config = BenchmarkDynamicConfig(
                    build_cmd=Command(
                        # $CC is replaced with clang command,
                        # $IN is replaced with benchmark path
                        # Following are linking flags (only one in this case).
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

        self._benchmarks = {
            f"{dataset_name}/mm": benchmark_from_file_contents(
                f"{dataset_name}/mm",
                self.preprocess(BENCHMARKS_PATH /"mm.txt"),
                benchmark_config
            ),
            f"{dataset_name}/mm1": benchmark_from_file_contents(
                f"{dataset_name}/mm1",
                self.preprocess(BENCHMARKS_PATH /"mm1.txt"),
                benchmark_config
            )
        }

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
