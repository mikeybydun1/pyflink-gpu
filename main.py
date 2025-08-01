from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.typeinfo import Types
from pyflink.datastream.functions import MapFunction
import torch


class MatMulMapFunction(MapFunction):
    def __init__(self):
        self._device = torch.device("cuda")

    def map(self, value):
        name, _ = value

        a = torch.randn(400, 400, device=self._device, dtype=torch.float32)
        b = torch.randn(400, 400, device=self._device, dtype=torch.float32)

        total = (a @ b).sum().item()
        print(f"Finished calculating matmul: {total}")
        return value


def main() -> None:
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    torch.manual_seed(42)

    records = [
        ("alice", 1),
        ("bob", 2),
        ("alice", 5),
        ("bob", 3),
        ("carol", 7),
    ]

    ds = env.from_collection(
        collection=records,
        type_info=Types.TUPLE([Types.STRING(), Types.INT()])
    )

    result = (
        ds
        .map(MatMulMapFunction(),
             output_type=Types.TUPLE([Types.STRING(), Types.FLOAT()]))
    )

    result.print()
    env.execute("tensor_matmul_per_record")


if __name__ == "__main__":
    main()

    import os, sys

    if sys.platform.startswith("win"):
        os._exit(0)
