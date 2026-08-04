# Owner(s): ["module: inductor"]

import ctypes
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from torch._inductor import cpp_builder
from torch._inductor.cpp_builder import CppBuilder, CppTorchDeviceOptions


class CppBuilderLibcxxTest(unittest.TestCase):
    def _create_fbpkg_layout(self, root: Path) -> tuple[Path, Path]:
        archive_dir = root / "lib" / "x86_64"
        archive_dir.mkdir(parents=True)
        for archive in ("libaoti_c++.a", "libaoti_c++abi.a"):
            (archive_dir / archive).touch()
        include_dir = root / "include"
        header_dir = include_dir / "c++" / "v1"
        header_dir.mkdir(parents=True)
        for header in ("array", "__config"):
            (header_dir / header).touch()
        return archive_dir, include_dir

    def test_resolves_and_stages_fbpkg_layout(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            archive_dir, include_dir = self._create_fbpkg_layout(root)
            with (
                mock.patch.dict(
                    "os.environ",
                    {
                        "LOWER_PKG_PATH": str(root / "ien.lower"),
                    },
                    clear=True,
                ),
                mock.patch.object(
                    cpp_builder.platform, "machine", return_value="x86_64"
                ),
                tempfile.TemporaryDirectory() as staging_dir,
            ):
                paths = cpp_builder._require_aoti_libcxx_paths()
                command = [
                    f"-L{archive_dir}",
                    f"-I{include_dir / 'c++' / 'v1'}",
                ]
                staged_command = cpp_builder._stage_aoti_libcxx_files(
                    command, staging_dir
                )

                self.assertEqual(str(archive_dir), paths.archive_dir)
                self.assertEqual(str(include_dir), paths.include_dir)
                self.assertIn(f"-L{staging_dir}", staged_command)
                self.assertIn(
                    f"-I{staging_dir}/aoti_libcxx_include/c++/v1",
                    staged_command,
                )
                self.assertTrue(Path(staging_dir, "libaoti_c++.a").is_file())
                self.assertTrue(Path(staging_dir, "libaoti_c++abi.a").is_file())

    def test_builds_and_loads_private_libcxx_dso(self) -> None:
        source = r"""
#include <cstddef>
#include <string>

extern "C" __attribute__((visibility("default")))
std::size_t aoti_libcxx_string_size(const char* value) {
  return std::string(value).size();
}
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "private_libcxx.cpp"
            source_path.write_text(source)
            builder = CppBuilder(
                name="private_libcxx",
                sources=str(source_path),
                output_dir=tmp_dir,
                BuildOption=CppTorchDeviceOptions(
                    aot_mode=True,
                    cpp_stdlib="libc++",
                    device_type="cpu",
                ),
            )

            builder.build()
            library_path = builder.get_target_file_path()
            library = ctypes.CDLL(library_path)
            string_size = library.aoti_libcxx_string_size
            string_size.argtypes = [ctypes.c_char_p]
            string_size.restype = ctypes.c_size_t

            self.assertEqual(5, string_size(b"aoti!"))
            dynamic_section = subprocess.check_output(
                ["readelf", "-d", library_path], text=True
            )
            self.assertNotIn("libstdc++.so", dynamic_section)
