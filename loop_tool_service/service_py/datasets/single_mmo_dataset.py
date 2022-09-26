from compiler_gym.datasets import Benchmark, BenchmarkUri, Dataset
from compiler_gym.util.runfiles_path import site_data_path
from typing import Iterable
from pathlib import Path
from loop_tool_service.paths import BENCHMARKS_PATH

from . import benchmark_from_file_contents
from compiler_gym.service.proto import BenchmarkDynamicConfig, Command


BENCHMARKS_PATH = BENCHMARKS_PATH/"loop_tool_dataset/mmo_perm"


class Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        dataset_name = "benchmark://single_mmo-v0"
        super().__init__(
            name=dataset_name,
            license="MIT",
            description="Single optimized matrix multiply 128 dataset",
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
            f"{dataset_name}/012345": benchmark_from_file_contents(
                f"{dataset_name}/012345",
                self.preprocess(BENCHMARKS_PATH /"012345.txt"),
                benchmark_config
            ),
            f"{dataset_name}/302451": benchmark_from_file_contents(
                f"{dataset_name}/302451",
                self.preprocess(BENCHMARKS_PATH /"302451.txt"),
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
