# Owner(s): ["oncall: distributed checkpointing"]

import json
import os
import tempfile
from unittest.mock import ANY

from safetensors.torch import load_file, save_file

import torch
from torch.distributed.checkpoint._hf_storage import (
    _metadata_fn,
    HuggingFaceHubReader,
    HuggingFaceHubWriter,
    SUFFIX,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.filesystem import _StorageInfo, FileSystem
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    MetadataIndex,
)
from torch.distributed.checkpoint.planner import LoadPlan, SavePlan
from torch.distributed.checkpoint.planner_helpers import (
    _create_read_items,
    _create_write_item_for_tensor,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.testing._internal.common_utils import run_tests, TestCase


class TestHfStorage(TestCase):

    def test_write_data_hf(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            writer = HuggingFaceHubWriter(
                path=path,
            )
            writer.fs = FileSystem()

            tensor0 = torch.rand(4)
            tensor1 = torch.rand(10)
            write_item_1 = _create_write_item_for_tensor("tensor_0", tensor0)
            write_item_2 = _create_write_item_for_tensor("tensor_1", tensor1)

            state_dict = {"tensor_0": tensor0, "tensor_1": tensor1}

            save_plan = SavePlan(
                [write_item_1, write_item_2],
                storage_data={"tensor_0": 1, "tensor_1": 1},
            )
            save_planner = DefaultSavePlanner()
            save_planner.set_up_planner(state_dict=state_dict)

            write_results = writer.write_data(save_plan, save_planner)

            write_results.wait()
            actual_write_results = write_results.value()

            expected_write_results = [
                WriteResult(
                    index=MetadataIndex(fqn="tensor_0", offset=None, index=None),
                    size_in_bytes=ANY,
                    storage_data=ANY,
                ),
                WriteResult(
                    index=MetadataIndex(fqn="tensor_1", offset=None, index=None),
                    size_in_bytes=ANY,
                    storage_data=ANY,
                ),
            ]

            self.assertEqual(
                actual_write_results[0].storage_data,
                expected_write_results[0].storage_data,
            )

            tensor_dict = load_file(os.path.join(path, "model-00001-of-00001" + SUFFIX))
            self.assertEqual(tensor_dict["tensor_0"], tensor0)
            self.assertEqual(tensor_dict["tensor_1"], tensor1)

    def test_read_data_hf(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            reader = HuggingFaceHubReader(path=path)
            reader.fs = FileSystem()
            name = "tensor_0"
            file_name = "model-00001-of-00001"
            tensor_0 = torch.rand(4)
            save_file({name: tensor_0}, os.path.join(path, file_name))

            reader.set_up_storage_reader(
                Metadata(
                    state_dict_metadata={name: BytesStorageMetadata()},
                    storage_data={name: file_name},
                ),
                is_coordinator=True,
            )

            read_items = _create_read_items(name, BytesStorageMetadata(), file_name)
            load_plan = LoadPlan(read_items)
            load_planner = DefaultLoadPlanner()
            load_planner.set_up_planner(state_dict={name: torch.rand(4)})

            read_data = reader.read_data(load_plan, load_planner)
            read_data.wait()

            loaded_tensor = load_planner.original_state_dict[name]
            self.assertEqual(loaded_tensor, tensor_0)

    def test_metadata_hf(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            file_name = "model-00001-of-00001"
            write_results = [
                WriteResult(
                    index=MetadataIndex(fqn="tensor_0", offset=None, index=None),
                    size_in_bytes=100,
                    storage_data=_StorageInfo(
                        relative_path=file_name, offset=0, length=100
                    ),
                ),
                WriteResult(
                    index=MetadataIndex(fqn="tensor_1", offset=None, index=None),
                    size_in_bytes=100,
                    storage_data=_StorageInfo(
                        relative_path=file_name, offset=0, length=100
                    ),
                ),
            ]

            writer = HuggingFaceHubWriter(
                path=path,
            )
            writer.fs = FileSystem()
            writer.finish(
                Metadata(
                    state_dict_metadata={
                        "tensor_0": BytesStorageMetadata(),
                        "tensor_1": BytesStorageMetadata(),
                    }
                ),
                results=[write_results],
            )
            metadata_file = os.path.join(path, _metadata_fn)

            expected_metadata = {
                "metadata": {"total_size": 200},
                "weight_map": {
                    "tensor_0": "model-00001-of-00001",
                    "tensor_1": "model-00001-of-00001",
                },
            }
            with open(metadata_file) as f:
                metadata = json.load(f)
                self.assertEqual(metadata, expected_metadata)

            reader = HuggingFaceHubReader(path=path)
            reader.fs = FileSystem()
            metadata = reader.read_metadata()
            self.assertEqual(metadata.storage_data, expected_metadata["weight_map"])


if __name__ == "__main__":
    run_tests()
