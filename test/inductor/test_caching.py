# Owner(s): ["module: inductor"]
# pyre-strict
from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, wait
from contextlib import contextmanager
from functools import wraps
from itertools import combinations
from random import Random
from shutil import rmtree
from threading import Lock
from typing import Any, TYPE_CHECKING, Union
from typing_extensions import TypeVar
from unittest.mock import patch

from filelock import FileLock

import torch
from torch._inductor.runtime.caching import (
    config,
    context,
    exceptions,
    implementations as impls,
    interfaces,
    locks,
    Memoizer,
    PersistentMemoizer,
    utils,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path


set_caching_module_enabled = lambda enabled: patch.object(  # noqa: E731
    config, "IS_CACHING_MODULE_ENABLED", lambda: enabled
)
set_deterministic_caching_enabled = lambda enabled: patch.object(  # noqa: E731
    config, "IS_DETERMINISTIC_CACHING_ENABLED", lambda: enabled
)
set_strictly_pre_populated_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "STRICTLY_PRE_POPULATED_DETERMINISM", enabled
)
set_strictly_cached_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "STRICTLY_CACHED_DETERMINISM", enabled
)
set_local_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "LOCAL_DETERMINISM", enabled
)
set_global_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "GLOBAL_DETERMINISM", enabled
)


def patch_on_disk_cache_base_dir(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        default_base_dir = impls._OnDiskCacheImpl()._base_dir
        with patch.object(
            impls._OnDiskCacheImpl,
            "_base_dir",
            default_base_dir / f"{self.sub_dir()}/rng-{self.random_string[4:]}",
        ):
            return fn(self, *args, **kwargs)

    return wrapper


def patch_remote_cache_with_on_disk_cache(fn):
    impls._OnDiskCacheImpl.has_strong_consistency = True
    return patch.object(impls, "_RemoteCacheImpl", impls._OnDiskCacheImpl)(fn)


class TestMixin:
    impl_typenames: list[str] = [
        "_InMemoryCacheImpl",
        "_OnDiskCacheImpl",
    ]
    cls_id: int = Random().randint(0, 2**32)

    def impl_from_typename(self, impl_typename: str) -> impls._CacheImpl:
        return getattr(impls, impl_typename)()

    @property
    def random_string(self) -> str:
        return f"s-{Random().randint(0, 2**32)}"

    @property
    def random_bytes(self) -> bytes:
        return f"s-{Random().randint(0, 2**32)}".encode()


@instantiate_parametrized_tests
class ConfigTest(TestCase):
    FOO_THIS_VERSION: int = 0
    FOO_JK_NAME: str = "foo_jk_name"
    FOO_OSS_DEFAULT: bool = False
    FOO_ENV_VAR_OVERRIDE: str = "foo_env_var_override"
    FOO_ENV_VAR_OVERRIDE_LOCK_FPATH: str = f"/tmp/testing/{FOO_ENV_VAR_OVERRIDE}.lock"
    FOO_ENV_VAR_OVERRIDE_LOCK: FileLock = FileLock(FOO_ENV_VAR_OVERRIDE_LOCK_FPATH)

    @classmethod
    def setUpClass(cls) -> None:
        rmtree(cls.FOO_ENV_VAR_OVERRIDE_LOCK_FPATH, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(cls.FOO_ENV_VAR_OVERRIDE_LOCK_FPATH, ignore_errors=True)

    def assert_versioned_config(self, expected_enabled: bool) -> None:
        config._versioned_config.cache_clear()
        actual_enabled: bool = config._versioned_config(
            self.FOO_JK_NAME,
            self.FOO_THIS_VERSION,
            self.FOO_OSS_DEFAULT,
            env_var_override=self.FOO_ENV_VAR_OVERRIDE,
        )
        self.assertEqual(actual_enabled, expected_enabled)

    @parametrize("enabled", [True, False])
    def test_versioned_config_env_var_override(
        self,
        enabled: bool,
    ) -> None:
        """Test that environment variable overrides take precedence over other configuration sources.

        Verifies that when an environment variable override is set to "1" or "0",
        the _versioned_config function returns the corresponding boolean value
        regardless of other configuration settings.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(
                os.environ,
                {
                    self.FOO_ENV_VAR_OVERRIDE: "1" if enabled else "0",
                },
            ),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=False,
            ),
            patch.object(self, "FOO_OSS_DEFAULT", not enabled),
        ):
            self.assert_versioned_config(enabled)

    @parametrize("enabled", [True, False])
    def test_versioned_config_version_check(
        self,
        enabled: bool,
    ) -> None:
        """Test that _versioned_config responds correctly to version changes in Facebook environments.

        Verifies that when running in fbcode environments (is_fbcode=True), the configuration
        is enabled when the JustKnobs version matches the expected version, and disabled when
        the version differs. This ensures proper rollout control through version management.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=True,
            ),
            patch(
                "torch._utils_internal.justknobs_getval_int",
                return_value=self.FOO_THIS_VERSION + (-1 if enabled else 1),
            ),
        ):
            self.assert_versioned_config(enabled)

    @parametrize("enabled", [True, False])
    def test_versioned_config_oss_default(
        self,
        enabled: bool,
    ) -> None:
        """Test that _versioned_config uses OSS default values in non-Facebook environments.

        Verifies that when running in non-fbcode environments (is_fbcode=False) with no
        environment variable overrides, the configuration falls back to the OSS default
        value. This ensures proper behavior for open-source PyTorch distributions.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=False,
            ),
            patch.object(self, "FOO_OSS_DEFAULT", enabled),
        ):
            self.assert_versioned_config(enabled)

    def test_versioned_config_jk_failure(self) -> None:
        """Test that _versioned_config uses OSS default values in non-Facebook environments.

        Verifies that when running in non-fbcode environments (is_fbcode=False) with no
        environment variable overrides, the configuration falls back to the OSS default
        value. This ensures proper behavior for open-source PyTorch distributions.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=True,
            ),
            patch(
                "torch._utils_internal.justknobs_getval_int",
                return_value=0,
            ),
        ):
            self.assert_versioned_config(False)


@instantiate_parametrized_tests
class ContextTest(TestCase):
    def isolation_schema_from_forms_of_context_selected(
        self,
        runtime_forms_of_context_selected: Sequence[str],
        compile_forms_of_context_selected: Sequence[str],
    ) -> context.IsolationSchema:
        return context.IsolationSchema(
            runtime_context={
                form_of_context: form_of_context
                in set(runtime_forms_of_context_selected)
                for form_of_context in context._RuntimeContext.forms_of_context()
            },
            compile_context={
                form_of_context: form_of_context
                in set(compile_forms_of_context_selected)
                for form_of_context in context._CompileContext.forms_of_context()
            },
        )

    @parametrize(
        "runtime_forms_of_context_selected",
        [(), *list(combinations(context._RuntimeContext.forms_of_context(), 2))],
    )
    @parametrize(
        "compile_forms_of_context_selected",
        [(), *list(combinations(context._CompileContext.forms_of_context(), 2))],
    )
    def test_selected_isolation_context(
        self,
        runtime_forms_of_context_selected: Sequence[str],
        compile_forms_of_context_selected: Sequence[str],
    ) -> None:
        """
        Tests that isolation context generation works correctly for specific combinations
        of runtime and compile context forms.

        Verifies that the _isolation_context function properly creates isolation contexts
        based on the selected forms of runtime and compile context, ensuring that only
        the specified context forms are included in the resulting isolation context.
        """
        ischema: context.IsolationSchema = (
            self.isolation_schema_from_forms_of_context_selected(
                runtime_forms_of_context_selected, compile_forms_of_context_selected
            )
        )

        self.assertEqual(
            context._isolation_context(ischema),
            {
                "runtime_context": {
                    form_of_context: getattr(context._RuntimeContext, form_of_context)()
                    for form_of_context in runtime_forms_of_context_selected
                }
                or None,
                "compile_context": {
                    form_of_context: getattr(context._CompileContext, form_of_context)()
                    for form_of_context in compile_forms_of_context_selected
                }
                or None,
            },
        )

    @parametrize("all_runtime_context", [True, False])
    @parametrize("all_compile_context", [True, False])
    def test_all_or_none_isolation_context(
        self, all_runtime_context: bool, all_compile_context: bool
    ) -> None:
        """
        Tests isolation context generation when using all or no context forms.

        Verifies that the isolation context correctly includes all forms of context
        when set to True, or excludes all forms when set to False, for both
        runtime and compile contexts.
        """
        ischema: context.IsolationSchema = context.IsolationSchema(
            runtime_context=all_runtime_context, compile_context=all_compile_context
        )
        self.assertEqual(
            context._isolation_context(ischema),
            {
                "runtime_context": {
                    form_of_context: getattr(context._RuntimeContext, form_of_context)()
                    for form_of_context in context._RuntimeContext.forms_of_context()
                }
                if all_runtime_context
                else None,
                "compile_context": {
                    form_of_context: getattr(context._CompileContext, form_of_context)()
                    for form_of_context in context._CompileContext.forms_of_context()
                }
                if all_compile_context
                else None,
            },
        )

    def test_isolation_key_is_distinct(self) -> None:
        """
        Tests that different combinations of runtime and compile context forms
        generate unique isolation keys.

        Verifies that each possible combination of context forms produces a distinct
        isolation key, ensuring no collisions occur between different contexts.
        """
        ikeys: set[str] = set()
        for num_runtime_forms_of_context_selected in range(
            len(context._RuntimeContext.forms_of_context())
        ):
            for num_compile_forms_of_context_selected in range(
                len(context._CompileContext.forms_of_context())
            ):
                for runtime_forms_of_context_selected in combinations(
                    context._RuntimeContext.forms_of_context(),
                    num_runtime_forms_of_context_selected,
                ):
                    for compile_forms_of_context_selected in combinations(
                        context._CompileContext.forms_of_context(),
                        num_compile_forms_of_context_selected,
                    ):
                        ischema: context.IsolationSchema = (
                            self.isolation_schema_from_forms_of_context_selected(
                                runtime_forms_of_context_selected,
                                compile_forms_of_context_selected,
                            )
                        )
                        ikey: str = context._isolation_key(ischema)
                        self.assertFalse(ikey in ikeys)
                        ikeys.add(ikey)

    def test_isolation_key_is_repeatable(self) -> None:
        """
        Tests that calling the isolation key function multiple times with the same
        parameters produces the same result.

        Verifies that the isolation key generation is deterministic and consistent
        across multiple invocations with identical inputs.
        """
        self.assertEqual(context._isolation_key(), context._isolation_key())

    def test_select_runtime_context_matches_forms_of_context(self) -> None:
        """
        Tests that the selected runtime context matches the forms of context.

        Verifies that the selected runtime context includes only the forms of context
        specified in the isolation schema, ensuring that the isolation context is
        properly selected and configured.
        """
        self.assertEqual(
            set(context.SelectedRuntimeContext.__required_keys__),
            set(context._RuntimeContext.forms_of_context()),
        )

    def test_select_compile_context_matches_forms_of_context(self) -> None:
        """
        Tests that the selected compile context matches the forms of context.

        Verifies that the selected compile context includes only the forms of context
        specified in the isolation schema, ensuring that the isolation context is
        properly selected and configured.
        """
        self.assertEqual(
            set(context.SelectedCompileContext.__required_keys__),
            set(context._CompileContext.forms_of_context()),
        )


@instantiate_parametrized_tests
class ExceptionsTest(TestCase):
    exception_typenames: list[str] = [
        "CacheError",
        "SystemError",
        "LockTimeoutError",
        "FileLockTimeoutError",
        "UserError",
        "KeyEncodingError",
        "ValueEncodingError",
        "ValueDecodingError",
    ]

    @parametrize("exception_typename", exception_typenames)
    def test_exception_is_CacheError(self, exception_typename: str) -> None:
        """Test that all custom cache exceptions inherit from the base CacheError class.

        Verifies that every exception type defined in the caching exceptions module
        is properly derived from CacheError, ensuring consistent exception hierarchy
        and enabling unified exception handling throughout the caching system.
        """
        self.assertTrue(
            issubclass(getattr(exceptions, exception_typename), exceptions.CacheError)
        )

    def test_exception_other(self) -> None:
        """
        Test the inheritance relationships among custom cache exception classes.

        Verifies that the exception classes in the caching exceptions module have the correct
        subclass relationships, ensuring the exception hierarchy is as intended. This includes
        checks for both direct and indirect inheritance between base and derived exception types.
        """
        self.assertTrue(issubclass(exceptions.SystemError, exceptions.CacheError))
        self.assertTrue(issubclass(exceptions.LockTimeoutError, exceptions.SystemError))
        self.assertTrue(
            issubclass(exceptions.FileLockTimeoutError, exceptions.SystemError)
        )
        self.assertTrue(issubclass(exceptions.UserError, exceptions.CacheError))
        self.assertTrue(issubclass(exceptions.KeyEncodingError, exceptions.UserError))
        self.assertTrue(issubclass(exceptions.ValueEncodingError, exceptions.UserError))
        self.assertTrue(issubclass(exceptions.ValueDecodingError, exceptions.UserError))


@instantiate_parametrized_tests
class ImplementationsTest(TestMixin, TestCase):
    @classmethod
    def sub_dir(cls) -> str:
        return f"testing-impls-instance-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir, ignore_errors=True
        )

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir, ignore_errors=True
        )

    def assert_key_in(self, key: str, impl: impls._CacheImpl) -> None:
        self.assertTrue(impl.get(key) is not None)

    def assert_key_not_in(self, key: str, impl: impls._CacheImpl) -> None:
        self.assertTrue(impl.get(key) is None)

    def assert_key_value_inserted_in(
        self, key: str, value: Any, impl: impls._CacheImpl
    ) -> None:
        self.assertTrue(impl.insert(key, value))

    def assert_key_value_not_inserted_in(
        self, key: str, value: Any, impl: impls._CacheImpl
    ) -> None:
        self.assertFalse(impl.insert(key, value))

    def assert_key_has_value_in(
        self, key: str, value: Any, impl: impls._CacheImpl
    ) -> None:
        self.assertTrue(((get := impl.get(key)) is not None) and (get.value == value))

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_get(self, impl_typename: str) -> None:
        """Test cache get operation returns cache miss for non-existent keys.

        Verifies that both in-memory and on-disk cache implementations correctly
        handle get operations for keys that do not exist in the cache. This test
        ensures that the cache properly returns a cache miss (hit=False) when
        attempting to retrieve a non-existent key.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            self.assert_key_not_in(self.random_string, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_insert(self, impl_typename: str) -> None:
        """Test cache insert operation successfully stores and retrieves key-value pairs.

        Verifies that both in-memory and on-disk cache implementations correctly
        handle insert operations for new key-value pairs. This test ensures that:
        1. Keys initially don't exist in the cache (cache miss)
        2. Insert operations succeed for new keys
        3. The stored value can be retrieved correctly after insertion

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            key: str = self.random_string
            self.assert_key_not_in(key, impl)
            value: bytes = self.random_bytes
            self.assert_key_value_inserted_in(key, value, impl)
            self.assert_key_has_value_in(key, value, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_insert_will_not_overwrite(self, impl_typename: str) -> None:
        """Test cache insert operation does not overwrite existing keys.

        Verifies that both in-memory and on-disk cache implementations correctly
        handle insert operations for keys that already exist in the cache. This test
        ensures that:
        1. Keys initially don't exist in the cache (cache miss)
        2. First insert operation succeeds for new keys
        3. Subsequent insert operations with the same key fail (inserted=False)
        4. The original value is preserved and not overwritten

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            key: str = self.random_string
            self.assert_key_not_in(key, impl)
            value: bytes = self.random_bytes
            self.assert_key_value_inserted_in(key, value, impl)
            self.assert_key_value_not_inserted_in(key, self.random_bytes, impl)
            self.assert_key_has_value_in(key, value, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_value_encoding(self, impl_typename: str) -> None:
        """Test that in-memory cache can store any value type.

        Verifies that in-memory cache implementations can store arbitrary values
        including non-serializable ones (such as lambda functions) since they don't
        require serialization. On-disk caches now expect bytes, so they skip this test.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._InMemoryCacheImpl):
                key: str = self.random_string
                value = lambda: None  # noqa: E731
                self.assert_key_value_inserted_in(key, value, impl)
                self.assert_key_has_value_in(key, value, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_value_decoding(self, impl_typename: str) -> None:
        """Test that on-disk cache implementations return raw bytes from storage.

        Verifies that on-disk cache implementations return raw bytes from disk
        without attempting to unpickle them, as values are now expected to be
        stored as bytes. This test writes raw bytes to a cache file and confirms
        they are returned as-is.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._OnDiskCacheImpl):
                key: str = self.random_string
                self.assert_key_not_in(key, impl)
                fpath: Path = impl._fpath_from_key(key)
                test_bytes: bytes = b"foo"
                with open(fpath, "xb") as fp:
                    impl._write_version_header(fp)
                    fp.write(test_bytes)
                self.assert_key_has_value_in(key, test_bytes, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_version_mismatch(self, impl_typename: str) -> None:
        """Test that on-disk cache implementations properly handle version mismatches.

        Verifies that on-disk cache implementations correctly handle cached data when
        the cache version changes. This test ensures that:
        1. Data can be stored and retrieved with the current version
        2. When version changes, previously cached data becomes inaccessible (cache miss)
        3. New data can be stored with the new version
        4. After version change, old cached data remains inaccessible

        This version checking mechanism prevents corruption and compatibility issues
        when cache formats change between software versions. Only applies to on-disk
        implementations since in-memory caches don't persist across version changes.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._OnDiskCacheImpl):
                key: str = self.random_string
                self.assert_key_not_in(key, impl)
                value: bytes = self.random_bytes
                self.assert_key_value_inserted_in(key, value, impl)
                self.assert_key_has_value_in(key, value, impl)
                with patch.object(
                    impls._OnDiskCacheImpl, "_version", impl._version + 1
                ):
                    self.assert_key_not_in(key, impl)
                    self.assert_key_value_inserted_in(key, value, impl)
                    self.assert_key_has_value_in(key, value, impl)
                self.assert_key_not_in(key, impl)
                self.assert_key_value_inserted_in(key, value, impl)
                self.assert_key_has_value_in(key, value, impl)


@instantiate_parametrized_tests
class LocksTest(TestMixin, TestCase):
    T = TypeVar("T")

    @contextmanager
    def executor(self) -> Generator[ThreadPoolExecutor, None, None]:
        executor: ThreadPoolExecutor = ThreadPoolExecutor()
        try:
            yield executor
        finally:
            executor.shutdown()

    def is_lock(self, lock_or_flock: Union[Lock, FileLock]) -> bool:
        return hasattr(lock_or_flock, "locked")

    def is_flock(self, lock_or_flock: Union[Lock, FileLock]) -> bool:
        return hasattr(lock_or_flock, "is_locked")

    def lock_or_flock_locked(self, lock_or_flock: Union[Lock, FileLock]) -> bool:
        if self.is_lock(lock_or_flock):
            return lock_or_flock.locked()
        elif self.is_flock(lock_or_flock):
            return lock_or_flock.is_locked
        else:
            raise NotImplementedError

    def test_BLOCKING(self) -> None:
        self.assertEqual(locks._BLOCKING, -1.0)

    def test_NON_BLOCKING(self) -> None:
        self.assertEqual(locks._NON_BLOCKING, 0.0)

    def test_BLOCKING_WITH_TIMEOUT(self) -> None:
        self.assertGreater(locks._BLOCKING_WITH_TIMEOUT, 0.0)

    @patch.object(locks, "_BLOCKING_WITH_TIMEOUT", 1.0)
    @patch.object(locks, "_DEFAULT_TIMEOUT", 1.0)
    @parametrize("lock_typename", ["Lock", "FileLock"])
    @parametrize("lock_timeout", ["BLOCKING", "NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"])
    @parametrize("acquisition_mode", ["safe", "unsafe"])
    @parametrize("release", ["unlocked", "never", "before_timeout", "after_timeout"])
    def test_acquire_with_timeout(
        self,
        lock_typename: str,
        lock_timeout: str,
        acquisition_mode: str,
        release: str,
    ) -> None:
        """Test lock acquisition behavior with various timeout configurations and release scenarios.

        This comprehensive test verifies the lock acquisition functionality for both threading.Lock
        and FileLock objects across different timeout modes, acquisition patterns, and release timings.
        The test validates proper exception handling, timeout behavior, and correct lock state management.

        Test parameters:
        - lock_typename: Tests both "Lock" (threading.Lock) and "FileLock" (filelock.FileLock) types
        - lock_timeout: Tests "BLOCKING", "NON_BLOCKING", and "BLOCKING_WITH_TIMEOUT" modes
        - acquisition_mode: Tests both "safe" (context manager) and "unsafe" (manual) acquisition
        - release: Tests "unlocked", "never", "before_timeout", and "after_timeout" scenarios

        The test ensures that:
        - Safe acquisition properly manages lock lifecycle through context managers
        - Unsafe acquisition requires manual release and behaves correctly
        - Timeout exceptions are raised appropriately for different timeout configurations
        - Lock states are correctly maintained throughout acquisition and release cycles
        - Different lock types (Lock vs FileLock) behave consistently with their respective APIs
        """

        def inner(lock_or_flock: Union[Lock, FileLock], timeout: int) -> None:
            if self.is_lock(lock_or_flock):
                lock: Lock = lock_or_flock
                if acquisition_mode == "safe":
                    with locks._acquire_lock_with_timeout(lock, timeout=timeout):
                        self.assertTrue(self.lock_or_flock_locked(lock))
                elif acquisition_mode == "unsafe":
                    locks._unsafe_acquire_lock_with_timeout(lock, timeout=timeout)
                    self.assertTrue(self.lock_or_flock_locked(lock))
                    lock.release()
                else:
                    raise NotImplementedError
            elif self.is_flock(lock_or_flock):
                flock: FileLock = lock_or_flock
                if acquisition_mode == "safe":
                    with locks._acquire_flock_with_timeout(flock, timeout=timeout):
                        self.assertTrue(self.lock_or_flock_locked(flock))
                elif acquisition_mode == "unsafe":
                    locks._unsafe_acquire_flock_with_timeout(flock, timeout=timeout)
                    self.assertTrue(self.lock_or_flock_locked(flock))
                    flock.release()
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            self.assertFalse(self.lock_or_flock_locked(lock_or_flock))

        assert lock_typename in ["Lock", "FileLock"]
        flock_fpath: Path = (
            impls._OnDiskCacheImpl()._cache_dir
            / f"testing-locks-instance-{self.random_string}.lock"
        )
        lock_or_flock: Union[Lock, FileLock] = (
            Lock() if lock_typename == "Lock" else FileLock(str(flock_fpath))
        )
        lock_exception_type: type = (
            exceptions.LockTimeoutError
            if lock_typename == "Lock"
            else exceptions.FileLockTimeoutError
        )

        if release == "unlocked":
            self.assertFalse(self.lock_or_flock_locked(lock_or_flock))
        elif release in ["never", "before_timeout", "after_timeout"]:
            self.assertTrue(lock_or_flock.acquire(timeout=locks._NON_BLOCKING))
            self.assertTrue(self.lock_or_flock_locked(lock_or_flock))
        else:
            raise NotImplementedError

        with self.executor() as executor:
            assert lock_timeout in ["BLOCKING", "NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"]
            lock_or_flock_future: Future[None] = executor.submit(
                inner,
                lock_or_flock,
                timeout={
                    "BLOCKING": locks._BLOCKING,
                    "NON_BLOCKING": locks._NON_BLOCKING,
                    "BLOCKING_WITH_TIMEOUT": locks._BLOCKING_WITH_TIMEOUT,
                }[lock_timeout],
            )

            if release == "unlocked":
                self.assertIsNone(lock_or_flock_future.result())
            elif release == "never":
                wait([lock_or_flock_future], timeout=(locks._BLOCKING_WITH_TIMEOUT * 2))
                if lock_timeout == "BLOCKING":
                    with self.assertRaises(TimeoutError):
                        lock_or_flock_future.result(
                            timeout=locks._BLOCKING_WITH_TIMEOUT
                        )
                elif lock_timeout in ["NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"]:
                    with self.assertRaises(lock_exception_type):
                        lock_or_flock_future.result()
                else:
                    raise NotImplementedError
                lock_or_flock.release()
            elif release == "before_timeout":
                wait([lock_or_flock_future], timeout=(locks._BLOCKING_WITH_TIMEOUT / 2))
                lock_or_flock.release()
                if lock_timeout in ["BLOCKING", "BLOCKING_WITH_TIMEOUT"]:
                    self.assertIsNone(lock_or_flock_future.result())
                elif lock_timeout == "NON_BLOCKING":
                    with self.assertRaises(lock_exception_type):
                        lock_or_flock_future.result()
                else:
                    raise NotImplementedError
            elif release == "after_timeout":
                wait([lock_or_flock_future], timeout=(locks._BLOCKING_WITH_TIMEOUT * 2))
                lock_or_flock.release()
                if lock_timeout == "BLOCKING":
                    self.assertIsNone(lock_or_flock_future.result())
                elif lock_timeout in ["NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"]:
                    with self.assertRaises(lock_exception_type):
                        lock_or_flock_future.result()
                else:
                    raise NotImplementedError

        flock_fpath.unlink(missing_ok=True)

    @patch.object(locks, "_BLOCKING_WITH_TIMEOUT", 1)
    @patch.object(locks, "_DEFAULT_TIMEOUT", 1)
    @parametrize(
        "impl_typename_combos",
        list(combinations(TestMixin.impl_typenames, 1))
        + list(combinations(TestMixin.impl_typenames, 2)),
    )
    def test_acquire_many_impl_locks_with_timeout(
        self,
        impl_typename_combos: tuple[str, ...],
    ) -> None:
        impls: list[impls._CacheImpl] = []
        for impl_typename in impl_typename_combos:
            impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
            impls.append(impl)

        with locks._acquire_many_impl_locks_with_timeout(*impls):
            for impl in impls:
                if hasattr(impl, "_lock"):
                    self.assertTrue(impl._lock.locked())
                elif hasattr(impl, "_flock"):
                    self.assertTrue(impl._flock.is_locked)

        for impl in impls:
            if hasattr(impl, "_lock"):
                self.assertFalse(impl._lock.locked())
            elif hasattr(impl, "_flock"):
                self.assertFalse(impl._flock.is_locked)


@instantiate_parametrized_tests
class UtilsTest(TestMixin, TestCase):
    def test_lru_cache(self) -> None:
        """Test that the LRU cache decorator works correctly with various argument types.

        Verifies that the _lru_cache decorator properly caches function results
        and handles different types of arguments including integers, floats, strings,
        and keyword arguments. Tests that cached calls return identical results
        to non-cached calls with proper argument preservation.
        """

        @utils._lru_cache
        def foo(*args, **kwargs):
            return args, kwargs

        self.assertEqual(
            foo(0),
            (
                (0,),
                {},
            ),
        )
        self.assertEqual(
            foo(0.0),
            (
                (0.0,),
                {},
            ),
        )
        self.assertEqual(
            foo("foo"),
            (
                ("foo",),
                {},
            ),
        )
        self.assertEqual(
            foo("foo", bar="bar"),
            (
                ("foo",),
                {"bar": "bar"},
            ),
        )


@instantiate_parametrized_tests
class InterfacesTest(TestMixin, TestCase):
    """Test class for Memoizer and PersistentMemoizer interfaces."""

    @classmethod
    def sub_dir(cls) -> str:
        return f"testing-interfaces-instance-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    # ============= Memoizer Tests =============

    @set_caching_module_enabled(True)
    def test_memoizer_record_caches_result(self) -> None:
        """Test that Memoizer.record() caches function results.

        Verifies that when a function is decorated with record(), its result
        is cached and can be retrieved later.
        """
        # Setup: create a memoizer and a function that tracks call count
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice with the same argument
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called twice (record always executes)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    @set_caching_module_enabled(True)
    def test_memoizer_replay_retrieves_cached_result(self) -> None:
        """Test that Memoizer.replay() retrieves cached results without executing the function.

        Verifies that when a function is decorated with replay(), it retrieves
        results from cache without executing the original function.
        """
        # Setup: create a memoizer, record a result, then try to replay it
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: record a result first
        compute(5)
        self.assertEqual(call_count, 1)

        # Create a replay function for the same computation
        @memoizer.replay()
        def compute_replay(x: int) -> int:
            # This should never be called
            raise AssertionError("Function should not be executed during replay")

        # Assert: replay retrieves the cached result without calling the function
        result = compute_replay(5)
        self.assertEqual(result, 10)
        self.assertEqual(call_count, 1)  # No additional calls

    @set_caching_module_enabled(True)
    def test_memoizer_replay_raises_on_cache_miss(self) -> None:
        """Test that Memoizer.replay() raises KeyError on cache miss.

        Verifies that when replay() is called with arguments that have no cached
        result, it raises a KeyError.
        """
        # Setup: create a memoizer with replay decorator
        memoizer = Memoizer()

        @memoizer.replay()
        def compute(x: int) -> int:
            return x * 2

        # Execute & Assert: replay raises KeyError for uncached arguments
        with self.assertRaises(KeyError):
            compute(5)

    @set_caching_module_enabled(True)
    def test_memoizer_memoize_caches_and_retrieves(self) -> None:
        """Test that Memoizer.memoize() caches on first call and retrieves on subsequent calls.

        Verifies that memoize() combines record and replay functionality:
        - First call executes the function and caches the result
        - Subsequent calls retrieve from cache without executing
        """
        # Setup: create a memoizer and a function that tracks call count
        memoizer = Memoizer()
        call_count = 0

        @memoizer.memoize()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice with the same argument
        result1 = compute(5)
        self.assertEqual(call_count, 1)  # Function called on first invocation

        result2 = compute(5)
        self.assertEqual(call_count, 1)  # Function not called on second invocation

        # Assert: both calls return the same result
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    @set_caching_module_enabled(False)
    def test_memoizer_record_disabled_returns_original_function(self) -> None:
        """Test that Memoizer.record() returns original function when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, record()
        returns the original function without any caching behavior.
        """
        # Setup: create a memoizer with caching disabled
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    @set_caching_module_enabled(False)
    def test_memoizer_replay_disabled_always_raises(self) -> None:
        """Test that Memoizer.replay() always raises KeyError when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, replay()
        always raises KeyError regardless of what's in the cache.
        """
        # Setup: create a memoizer with caching disabled
        memoizer = Memoizer()

        @memoizer.replay()
        def compute(x: int) -> int:
            return x * 2

        # Execute & Assert: replay always raises KeyError when disabled
        with self.assertRaises(KeyError) as cm:
            compute(5)
        self.assertIn("Caching is disabled", str(cm.exception))

    @set_caching_module_enabled(False)
    def test_memoizer_memoize_disabled_returns_original_function(self) -> None:
        """Test that Memoizer.memoize() returns original function when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, memoize()
        returns the original function without any caching behavior.
        """
        # Setup: create a memoizer with caching disabled
        memoizer = Memoizer()
        call_count = 0

        @memoizer.memoize()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    # ============= PersistentMemoizer Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_record_caches_to_both(self) -> None:
        """Test that PersistentMemoizer.record() caches to both memory and disk.

        Verifies that when a function is decorated with record(), its result
        is cached in both the in-memory cache and the on-disk cache.
        """
        # Setup: create a persistent memoizer
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function
        result = compute(5)

        # Assert: result is correct and cached in memory
        self.assertEqual(result, 10)
        self.assertEqual(call_count, 1)

        # Verify memory cache has the result as tuple (encoded_params, encoded_result)
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)
        # Cache now stores (encoded_params, encoded_result) tuple
        encoded_params, encoded_result = memory_hit.value
        self.assertEqual(encoded_result, 10)
        self.assertEqual(encoded_params, {"args": (5,), "kwargs": {}})

        # Verify disk cache has the result (pickled)
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNotNone(disk_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_replay_checks_memory_then_disk(self) -> None:
        """Test that PersistentMemoizer.replay() checks memory first, then disk.

        Verifies that replay() uses a two-level cache strategy:
        1. Check memory cache first (fast)
        2. Fall back to disk cache on memory miss
        3. Populate memory cache from disk on disk hit
        """
        # Setup: create a persistent memoizer and store only to disk
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())

        # Store a value directly to disk cache only (as tuple format)
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        import pickle

        # Cache now stores (encoded_params, encoded_result) tuple
        cached_tuple = ({"args": (5,), "kwargs": {}}, 10)
        pickled_value = pickle.dumps(cached_tuple)
        persistent._disk_cache.insert(cache_key, pickled_value)

        # Verify it's not in memory cache yet
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNone(memory_hit)

        # Create a replay function
        @persistent.replay()
        def compute(x: int) -> int:
            raise AssertionError("Function should not be executed during replay")

        # Execute: replay retrieves from disk and populates memory
        result = compute(5)

        # Assert: result is correct
        self.assertEqual(result, 10)

        # Verify memory cache was populated from disk
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)
        # Memory cache should now contain the tuple
        _, encoded_result = memory_hit.value
        self.assertEqual(encoded_result, 10)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_memoize_two_level_caching(self) -> None:
        """Test that PersistentMemoizer.memoize() uses two-level caching.

        Verifies that memoize() combines two-level caching behavior:
        - First call executes and caches to both memory and disk
        - Second call (same process) retrieves from memory
        - After clearing memory, retrieves from disk
        """
        # Setup: create a persistent memoizer
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.memoize()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: first call - cache miss, executes function
        result1 = compute(5)
        self.assertEqual(call_count, 1)
        self.assertEqual(result1, 10)

        # Second call - memory cache hit
        result2 = compute(5)
        self.assertEqual(call_count, 1)  # No additional execution
        self.assertEqual(result2, 10)

        # Clear memory cache to simulate a new process
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        persistent._memoizer._cache = impls._InMemoryCacheImpl()

        # Third call - memory miss, disk hit, populates memory
        result3 = compute(5)
        self.assertEqual(call_count, 1)  # Still no additional execution
        self.assertEqual(result3, 10)

        # Verify memory cache was repopulated
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_persistent_memoizer_record_disabled(self) -> None:
        """Test that PersistentMemoizer.record() returns original function when disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, record()
        returns the original function without any caching to memory or disk.
        """
        # Setup: create a persistent memoizer with caching disabled
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

        # Verify nothing was cached
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNone(memory_hit)
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNone(disk_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_persistent_memoizer_replay_disabled(self) -> None:
        """Test that PersistentMemoizer.replay() always raises when disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, replay()
        always raises KeyError.
        """
        # Setup: create a persistent memoizer with caching disabled
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())

        @persistent.replay()
        def compute(x: int) -> int:
            return x * 2

        # Execute & Assert: replay always raises KeyError when disabled
        with self.assertRaises(KeyError) as cm:
            compute(5)
        self.assertIn("Caching is disabled", str(cm.exception))

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_persistent_memoizer_memoize_disabled(self) -> None:
        """Test that PersistentMemoizer.memoize() returns original function when disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, memoize()
        returns the original function without any caching behavior.
        """
        # Setup: create a persistent memoizer with caching disabled
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.memoize()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    # ============= Cache Dump Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dumps_cache_to_json_on_exit(self) -> None:
        """Test that Memoizer dumps cache to JSON file with correct format.

        Verifies that the cache dump creates a JSON file at the expected path
        with the correct structure including "cache_size" and "cache_entries".
        """
        import json
        import os
        import tempfile

        # Setup: Create a temporary file path for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name

        try:
            # Setup: Create a memoizer with custom cache filepath
            memoizer = interfaces.Memoizer()
            memoizer._SHARED_CACHE_FILEPATH = test_filepath

            # Add some entries to the cache
            memoizer._cache.insert("key1", ({"param": "value1"}, True))
            memoizer._cache.insert("key2", ({"param": "value2"}, False))

            # Execute: Manually call _dump_cache_to_json (simulating program exit)
            memoizer._dump_cache_to_json()

            # Assert: Verify the JSON file was created and has correct structure
            self.assertTrue(os.path.exists(test_filepath))

            with open(test_filepath) as f:
                data = json.load(f)

            # Verify structure
            self.assertIn("cache_size", data)
            self.assertIn("cache_entries", data)
            self.assertEqual(data["cache_size"], 2)

            # Verify entries are formatted with "params" and "result"
            self.assertIn("key1", data["cache_entries"])
            self.assertIn("key2", data["cache_entries"])
            self.assertEqual(
                data["cache_entries"]["key1"],
                {"params": {"param": "value1"}, "result": True},
            )
            self.assertEqual(
                data["cache_entries"]["key2"],
                {"params": {"param": "value2"}, "result": False},
            )
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_is_additive(self) -> None:
        """Test that multiple Memoizer instances contribute additively to the same file.

        Verifies that when multiple memoizers dump their caches, entries are
        merged together in the same JSON file rather than overwriting.
        """
        import json
        import os
        import tempfile

        # Setup: Create a temporary file path for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name

        try:
            # Setup: Create first memoizer and dump
            memoizer1 = interfaces.Memoizer()
            memoizer1._SHARED_CACHE_FILEPATH = test_filepath
            memoizer1._cache.insert("key1", ({"param": "A"}, "result1"))
            memoizer1._dump_cache_to_json()

            # Setup: Create second memoizer and dump
            memoizer2 = interfaces.Memoizer()
            memoizer2._SHARED_CACHE_FILEPATH = test_filepath
            memoizer2._cache.insert("key2", ({"param": "B"}, "result2"))
            memoizer2._dump_cache_to_json()

            # Execute: Read the final JSON file
            with open(test_filepath) as f:
                data = json.load(f)

            # Assert: Both entries should be present (additive)
            self.assertEqual(data["cache_size"], 2)
            self.assertIn("key1", data["cache_entries"])
            self.assertIn("key2", data["cache_entries"])
            self.assertEqual(data["cache_entries"]["key1"]["result"], "result1")
            self.assertEqual(data["cache_entries"]["key2"]["result"], "result2")
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_skips_dump_when_cache_empty(self) -> None:
        """Test that Memoizer does not create a file when cache is empty.

        Verifies that the dump logic skips file creation when there's
        nothing to dump.
        """
        import os
        import tempfile

        # Setup: Create a temporary file path for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name

        # Remove the file so we can test it's not created
        os.unlink(test_filepath)

        try:
            # Setup: Create a memoizer with empty cache
            memoizer = interfaces.Memoizer()
            memoizer._SHARED_CACHE_FILEPATH = test_filepath

            # Execute: Call dump with empty cache
            memoizer._dump_cache_to_json()

            # Assert: File should not be created
            self.assertFalse(os.path.exists(test_filepath))
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_dumps_under_sub_key(self) -> None:
        """Test that PersistentMemoizer dumps cache under sub_dir key.

        Verifies that when sub_dir is non-empty, cache entries are nested
        under cache_entries[sub_dir] in the JSON file.
        """
        import json
        import os
        import tempfile
        from pathlib import Path

        # Setup: Create a temporary file path for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name

        try:
            # Setup: Create a PersistentMemoizer with sub_dir
            persistent_memoizer = interfaces.PersistentMemoizer(
                sub_dir=Path("test_subdir")
            )
            persistent_memoizer._memoizer._SHARED_CACHE_FILEPATH = test_filepath

            # Add entries to the internal memoizer
            persistent_memoizer._memoizer._cache.insert(
                "key1", ({"param": "nested"}, "nested_result")
            )

            # Execute: Dump the cache
            persistent_memoizer._memoizer._dump_cache_to_json()

            # Execute: Read the JSON file
            with open(test_filepath) as f:
                data = json.load(f)

            # Assert: Entry should be nested under sub_dir
            self.assertIn("cache_entries", data)
            self.assertIn("test_subdir", data["cache_entries"])
            self.assertIn("key1", data["cache_entries"]["test_subdir"])
            self.assertEqual(
                data["cache_entries"]["test_subdir"]["key1"]["result"],
                "nested_result",
            )
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_dumps_to_root_when_sub_dir_empty(self) -> None:
        """Test that PersistentMemoizer merges to root when sub_dir is empty.

        Verifies that when sub_dir is empty string, cache entries are merged
        directly into root cache_entries (not nested).
        """
        import json
        import os
        import tempfile

        # Setup: Create a temporary file path for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name

        try:
            # Setup: Create a PersistentMemoizer with empty sub_dir
            persistent_memoizer = interfaces.PersistentMemoizer(sub_dir="")
            persistent_memoizer._memoizer._SHARED_CACHE_FILEPATH = test_filepath

            # Add entries to the internal memoizer
            persistent_memoizer._memoizer._cache.insert(
                "root_key", ({"param": "root"}, "root_result")
            )

            # Execute: Dump the cache
            persistent_memoizer._memoizer._dump_cache_to_json()

            # Execute: Read the JSON file
            with open(test_filepath) as f:
                data = json.load(f)

            # Assert: Entry should be at root level, not nested
            self.assertIn("cache_entries", data)
            self.assertIn("root_key", data["cache_entries"])
            self.assertNotIn("", data["cache_entries"])  # No empty string key
            self.assertEqual(data["cache_entries"]["root_key"]["result"], "root_result")
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_multiple_persistent_memoizers_different_sub_dirs(self) -> None:
        """Test that multiple PersistentMemoizers with different sub_dirs coexist.

        Verifies that multiple PersistentMemoizer instances with different
        sub_dirs contribute to nested structures in the same JSON file.
        """
        import json
        import os
        import tempfile
        from pathlib import Path

        # Setup: Create a temporary file path for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name

        try:
            # Setup: Create first PersistentMemoizer with sub_dir="dir1"
            pm1 = interfaces.PersistentMemoizer(sub_dir=Path("dir1"))
            pm1._memoizer._SHARED_CACHE_FILEPATH = test_filepath
            pm1._memoizer._cache.insert("key1", ({"dir": "1"}, "result1"))
            pm1._memoizer._dump_cache_to_json()

            # Setup: Create second PersistentMemoizer with sub_dir="dir2"
            pm2 = interfaces.PersistentMemoizer(sub_dir=Path("dir2"))
            pm2._memoizer._SHARED_CACHE_FILEPATH = test_filepath
            pm2._memoizer._cache.insert("key2", ({"dir": "2"}, "result2"))
            pm2._memoizer._dump_cache_to_json()

            # Execute: Read the JSON file
            with open(test_filepath) as f:
                data = json.load(f)

            # Assert: Both sub_dirs should exist with their respective entries
            self.assertIn("dir1", data["cache_entries"])
            self.assertIn("dir2", data["cache_entries"])
            self.assertIn("key1", data["cache_entries"]["dir1"])
            self.assertIn("key2", data["cache_entries"]["dir2"])
            self.assertEqual(data["cache_entries"]["dir1"]["key1"]["result"], "result1")
            self.assertEqual(data["cache_entries"]["dir2"]["key2"]["result"], "result2")
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_cache_dump_formats_params_and_result_correctly(self) -> None:
        """Test that cache entries are formatted with 'params' and 'result' keys.

        Verifies that the JSON dump formats each cache entry as a dict with
        separate "params" and "result" keys for human readability.
        """
        import json
        import os
        import tempfile

        # Setup: Create a temporary file path for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name

        try:
            # Setup: Create a memoizer
            memoizer = interfaces.Memoizer()
            memoizer._SHARED_CACHE_FILEPATH = test_filepath

            # Add entry with complex params and result
            complex_params = {
                "tensor_shape": [128, 256],
                "dtype": "float32",
                "operation": "matmul",
            }
            complex_result = {"success": True, "time_ms": 42.5}

            memoizer._cache.insert("complex_key", (complex_params, complex_result))

            # Execute: Dump the cache
            memoizer._dump_cache_to_json()

            # Execute: Read the JSON file
            with open(test_filepath) as f:
                data = json.load(f)

            # Assert: Entry has correct structure
            entry = data["cache_entries"]["complex_key"]
            self.assertIn("params", entry)
            self.assertIn("result", entry)
            self.assertEqual(entry["params"], complex_params)
            self.assertEqual(entry["result"], complex_result)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    # ============= Cache Loading Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_loads_cache_from_dump_file(self) -> None:
        """Test that Memoizer loads cache entries from dump file on initialization.

        Verifies that when CACHE_DUMP_FILE_PATH is configured and the file exists,
        a new Memoizer instance pre-populates its in-memory cache with the dump contents.
        """
        import json
        import os
        import tempfile

        # Setup: Create a dump file with cache entries
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            dump_data = {
                "cache_size": 2,
                "cache_entries": {
                    "key1": {"params": {"x": 1}, "result": 10},
                    "key2": {"params": {"x": 2}, "result": 20},
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            # Setup: Configure CACHE_DUMP_FILE_PATH
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create a new Memoizer (should load from dump)
                memoizer = interfaces.Memoizer()

                # Assert: Cache was populated from dump file
                hit1 = memoizer._cache.get("key1")
                self.assertIsNotNone(hit1)
                _, result1 = hit1.value
                self.assertEqual(result1, 10)

                hit2 = memoizer._cache.get("key2")
                self.assertIsNotNone(hit2)
                _, result2 = hit2.value
                self.assertEqual(result2, 20)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_skips_loading_when_no_dump_file_configured(self) -> None:
        """Test that Memoizer skips loading when CACHE_DUMP_FILE_PATH is not set.

        Verifies that when no dump file path is configured, the Memoizer
        initializes with an empty cache without errors.
        """
        # Setup: Configure CACHE_DUMP_FILE_PATH to return None
        with patch.object(config, "CACHE_DUMP_FILE_PATH", return_value=None):
            # Execute: Create a new Memoizer
            memoizer = interfaces.Memoizer()

            # Assert: Cache is empty (not loaded from any file)
            self.assertEqual(len(memoizer._cache._memory), 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_handles_missing_dump_file_gracefully(self) -> None:
        """Test that Memoizer handles missing dump file gracefully.

        Verifies that when CACHE_DUMP_FILE_PATH points to a non-existent file,
        the Memoizer initializes with an empty cache without crashing.
        """
        # Setup: Configure path to non-existent file
        non_existent_path = "/tmp/this_file_does_not_exist_12345.json"

        with patch.object(
            config, "CACHE_DUMP_FILE_PATH", return_value=non_existent_path
        ):
            # Execute: Create a new Memoizer
            memoizer = interfaces.Memoizer()

            # Assert: Cache is empty (no crash)
            self.assertEqual(len(memoizer._cache._memory), 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_handles_corrupt_dump_file_gracefully(self) -> None:
        """Test that Memoizer handles corrupt dump file gracefully.

        Verifies that when the dump file contains invalid JSON, the Memoizer
        initializes with an empty cache without crashing.
        """
        import os
        import tempfile

        # Setup: Create a corrupt dump file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            tmp_file.write("{ this is not valid json")

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create a new Memoizer
                memoizer = interfaces.Memoizer()

                # Assert: Cache is empty (no crash, handled gracefully)
                self.assertEqual(len(memoizer._cache._memory), 0)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_loads_from_sub_key(self) -> None:
        """Test that PersistentMemoizer loads cache from sub_dir nested structure.

        Verifies that when sub_dir is set, the PersistentMemoizer loads entries
        from the nested cache_entries[sub_dir] structure.
        """
        import json
        import os
        import tempfile
        from pathlib import Path

        # Setup: Create a dump file with nested structure
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            dump_data = {
                "cache_size": 2,
                "cache_entries": {
                    "test_subdir": {
                        "nested_key1": {"params": {"x": 1}, "result": 100},
                        "nested_key2": {"params": {"x": 2}, "result": 200},
                    },
                    "other_subdir": {
                        "other_key": {"params": {"x": 3}, "result": 300},
                    },
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create PersistentMemoizer with sub_dir="test_subdir"
                pm = interfaces.PersistentMemoizer(sub_dir=Path("test_subdir"))

                # Assert: Cache loaded entries from test_subdir only
                hit1 = pm._memoizer._cache.get("nested_key1")
                self.assertIsNotNone(hit1)
                _, result1 = hit1.value
                self.assertEqual(result1, 100)

                hit2 = pm._memoizer._cache.get("nested_key2")
                self.assertIsNotNone(hit2)
                _, result2 = hit2.value
                self.assertEqual(result2, 200)

                # Assert: Did not load entries from other_subdir
                hit_other = pm._memoizer._cache.get("other_key")
                self.assertIsNone(hit_other)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_loads_from_root_when_sub_dir_empty(self) -> None:
        """Test that PersistentMemoizer loads from root when sub_dir is empty.

        Verifies that when sub_dir is empty string, entries are loaded from
        the root cache_entries level (not from any nested structure).
        """
        import json
        import os
        import tempfile

        # Setup: Create a dump file with mixed root and nested entries
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            dump_data = {
                "cache_size": 3,
                "cache_entries": {
                    "root_key1": {"params": {"x": 1}, "result": 10},
                    "root_key2": {"params": {"x": 2}, "result": 20},
                    "some_subdir": {
                        "nested_key": {"params": {"x": 3}, "result": 30},
                    },
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create PersistentMemoizer with empty sub_dir
                pm = interfaces.PersistentMemoizer(sub_dir="")

                # Assert: Loaded root-level entries
                hit1 = pm._memoizer._cache.get("root_key1")
                self.assertIsNotNone(hit1)
                _, result1 = hit1.value
                self.assertEqual(result1, 10)

                hit2 = pm._memoizer._cache.get("root_key2")
                self.assertIsNotNone(hit2)
                _, result2 = hit2.value
                self.assertEqual(result2, 20)

                # Assert: Did not load nested entries
                hit_nested = pm._memoizer._cache.get("nested_key")
                self.assertIsNone(hit_nested)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_replay_uses_preloaded_cache(self) -> None:
        """Test that memoizer replay successfully retrieves from preloaded cache.

        Verifies end-to-end workflow: load cache from dump file, then use
        replay to retrieve cached results without executing the function.
        """
        import json
        import os
        import tempfile

        # Setup: Create a dump file with a cached result
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            # Simulate a cache entry for compute(5) -> 10
            cache_key = interfaces._BaseMemoizer._make_key(None, 5)
            dump_data = {
                "cache_size": 1,
                "cache_entries": {
                    cache_key: {"params": {"args": (5,), "kwargs": {}}, "result": 10},
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create a memoizer (loads cache from dump)
                memoizer = interfaces.Memoizer()

                # Create a replay function
                @memoizer.replay()
                def compute(x: int) -> int:
                    raise AssertionError(
                        "Function should not be executed during replay"
                    )

                # Execute: Call replay (should use preloaded cache)
                result = compute(5)

                # Assert: Got cached result without executing function
                self.assertEqual(result, 10)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)


    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_dump_and_load_from_disk(self) -> None:
        """Test that _should_pad memoizer results can be dumped and loaded from disk.

        Verifies end-to-end workflow:
        1. Call _should_pad to populate the memoizer
        2. Dump the memoizer cache to disk
        3. Clear the in-memory cache
        4. Load cache entries from the dump file
        5. Call _should_pad again - should use loaded cached result without benchmarking
        """
        import json
        import os
        import tempfile
        from torch._inductor.fx_passes.pad_mm import (
            _should_pad,
            _should_pad_memoizer,
        )
        from unittest.mock import MagicMock

        # Create temp file for dump
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name

        # Remove to start clean
        os.unlink(test_filepath)

        try:
            # Create real tensors with unique shapes not used in other tests
            # to avoid disk cache hits from previous test runs
            mat1 = torch.randn(73, 147, dtype=torch.float32, device="cpu")
            mat2 = torch.randn(147, 293, dtype=torch.float32, device="cpu")
            match = MagicMock()

            # Clear and configure memoizer for testing
            # Use a consistent sub_key for both dump and load
            test_sub_key = "test_dump_load"
            _should_pad_memoizer._memoizer._cache = impls._InMemoryCacheImpl()
            _should_pad_memoizer._disk_cache = impls._OnDiskCacheImpl(
                sub_dir=self.sub_dir()
            )
            _should_pad_memoizer._memoizer._SHARED_CACHE_FILEPATH = test_filepath
            # Set sub_key to match what we'll use for loading
            _should_pad_memoizer._memoizer._sub_key = test_sub_key

            with (
                patch(
                    "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time"
                ) as mock_should_exclude,
                patch(
                    "torch._inductor.fx_passes.pad_mm.get_alignment_size"
                ) as mock_get_alignment_size,
                patch("torch._inductor.fx_passes.pad_mm.has_triton") as mock_has_triton,
                patch(
                    "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound"
                ) as mock_is_mm_compute_bound,
                patch(
                    "torch._inductor.fx_passes.pad_mm.get_cached_should_pad"
                ) as mock_get_cached_should_pad,
                patch(
                    "torch._inductor.fx_passes.pad_mm.get_do_bench"
                ) as mock_get_do_bench,
                patch(
                    "torch._inductor.fx_passes.pad_mm.get_cached_base_mm_benchmark_time"
                ) as mock_get_cached_base_mm,
                patch(
                    "torch._inductor.fx_passes.pad_mm.set_cached_base_mm_benchmark_time"
                ),
                patch("torch._inductor.fx_passes.pad_mm.should_pad") as mock_should_pad,
                patch(
                    "torch._inductor.fx_passes.pad_mm.is_padded_faster"
                ) as mock_is_padded_faster,
                patch("torch._inductor.config.force_shape_pad", False),
                patch("torch._inductor.config.deterministic", False),
                patch("torch._inductor.config.post_grad_fusion_options", {}),
            ):
                # Configure mocks
                mock_should_exclude.return_value = False
                mock_get_alignment_size.return_value = 8
                mock_has_triton.return_value = True
                mock_is_mm_compute_bound.return_value = True
                mock_get_cached_should_pad.return_value = None
                mock_get_cached_base_mm.return_value = None

                mock_do_bench = MagicMock(return_value=1.0)
                mock_get_do_bench.return_value = mock_do_bench

                mock_should_pad.return_value = True
                mock_is_padded_faster.return_value = True

                # Step 1: Call _should_pad to populate memoizer
                result1 = _should_pad(match, mat1, mat2, torch.ops.aten.mm)
                self.assertTrue(result1)
                first_call_benchmarks = mock_do_bench.call_count
                self.assertGreater(first_call_benchmarks, 0)

                # Step 2: Dump memoizer to disk
                _should_pad_memoizer._memoizer._dump_cache_to_json()
                self.assertTrue(os.path.exists(test_filepath))

                # Verify dump has content
                with open(test_filepath, "r") as f:
                    dump_data = json.load(f)
                self.assertGreater(dump_data["cache_size"], 0)

                # Step 3: Clear in-memory cache
                _should_pad_memoizer._memoizer._cache = impls._InMemoryCacheImpl()

                # Verify cache is empty
                self.assertEqual(
                    len(_should_pad_memoizer._memoizer._cache._memory), 0
                )

                # Step 4: Load from dump file
                with patch.object(
                    config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
                ):
                    # Create temp memoizer with the same sub_key to load entries
                    temp_memoizer = interfaces.PersistentMemoizer(
                        sub_dir=test_sub_key
                    )

                    # Copy loaded entries to the actual memoizer's cache
                    for key, value in temp_memoizer._memoizer._cache._memory.items():
                        _should_pad_memoizer._memoizer._cache.insert(
                            key, value
                        )

                # Verify cache was repopulated
                self.assertGreater(
                    len(_should_pad_memoizer._memoizer._cache._memory), 0
                )

                # Step 5: Call _should_pad again
                mock_do_bench.reset_mock()
                result2 = _should_pad(match, mat1, mat2, torch.ops.aten.mm)

                # Assert: Same result, no benchmarking (cache hit from loaded dump)
                self.assertTrue(result2)
                self.assertEqual(mock_do_bench.call_count, 0)

        finally:
            # Cleanup temp file
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

        from torch._inductor.fx_passes.pad_mm import _should_pad_params_encoder
        from unittest.mock import MagicMock

        # Create mock tensors with specific properties
        mat1 = MagicMock()
        mat1.shape = torch.Size([128, 256])
        mat1.stride.return_value = (256, 1)
        mat1.dtype = torch.float32

        mat2 = MagicMock()
        mat2.shape = torch.Size([256, 512])
        mat2.stride.return_value = (512, 1)
        mat2.dtype = torch.float16

        # Create mock match object
        match = MagicMock()

        # Mock the should_exclude_padding_time function
        with patch(
            "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time"
        ) as mock_should_exclude:
            mock_should_exclude.side_effect = lambda m, name: name == "mat1"

            # Execute: call the params encoder
            result = _should_pad_params_encoder(
                match, mat1, mat2, torch.ops.aten.mm, input=None
            )

        # Assert: verify the encoder extracted the correct information
        self.assertIsInstance(result, dict)
        self.assertEqual(result["mat1"]["shape"], (128, 256))
        self.assertEqual(result["mat1"]["stride"], (256, 1))
        self.assertEqual(result["mat1"]["dtype"], "torch.float32")
        self.assertEqual(result["mat2"]["shape"], (256, 512))
        self.assertEqual(result["mat2"]["stride"], (512, 1))
        self.assertEqual(result["mat2"]["dtype"], "torch.float16")
        self.assertEqual(result["op"], str(torch.ops.aten.mm))
        self.assertIsNone(result["input"])
        self.assertTrue(result["mat1_exclude_padding_time"])
        self.assertFalse(result["mat2_exclude_padding_time"])

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_bench_params_encoder_handles_input_tensor(self) -> None:
        """Test that the params encoder correctly handles optional input tensors.

        Verifies that when an input tensor is provided (as in addmm operations),
        the encoder extracts its information; when None, it stores None.
        """
        # Setup: Import the encoder and create mock objects
        from torch._inductor.fx_passes.pad_mm import _should_pad_params_encoder
        from unittest.mock import MagicMock

        mat1 = MagicMock()
        mat1.shape = torch.Size([128, 256])
        mat1.stride.return_value = (256, 1)
        mat1.dtype = torch.float32

        mat2 = MagicMock()
        mat2.shape = torch.Size([256, 512])
        mat2.stride.return_value = (512, 1)
        mat2.dtype = torch.float32

        input_tensor = MagicMock()
        input_tensor.shape = torch.Size([128])
        input_tensor.stride.return_value = (1,)
        input_tensor.dtype = torch.float32

        match = MagicMock()

        # Mock should_exclude_padding_time
        with patch(
            "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time"
        ) as mock_should_exclude:
            mock_should_exclude.return_value = False

            # Execute: call the encoder with an input tensor
            result = _should_pad_params_encoder(
                match, mat1, mat2, torch.ops.aten.addmm, input=input_tensor
            )

        # Assert: verify the input tensor info was extracted
        self.assertIsNotNone(result["input"])
        self.assertEqual(result["input"]["shape"], (128,))
        self.assertEqual(result["input"]["stride"], (1,))
        self.assertEqual(result["input"]["dtype"], "torch.float32")

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_bench_memoization_works(self) -> None:
        """Test that _should_pad_bench memoization caches results correctly.

        Verifies that when _should_pad_bench is called multiple times with the
        same encoded parameters, it returns cached results without re-executing
        the benchmarking logic.
        """
        # Setup: Import the memoized function and memoizer
        from torch._inductor.fx_passes.pad_mm import (
            _should_pad,
            _should_pad_memoizer,
        )
        from unittest.mock import MagicMock

        # Create real tensors (not FakeTensors or MagicMock) to avoid realize_tensor issues
        mat1 = torch.randn(65, 130, dtype=torch.float32, device="cpu")
        mat2 = torch.randn(130, 257, dtype=torch.float32, device="cpu")
        match = MagicMock()

        # Clear the memoizer caches to ensure clean state
        _should_pad_memoizer._memoizer._cache = impls._InMemoryCacheImpl()
        _should_pad_memoizer._disk_cache = impls._OnDiskCacheImpl(
            sub_dir=self.sub_dir()
        )

        # Mock all the dependencies that _should_pad uses
        with (
            patch(
                "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time"
            ) as mock_should_exclude,
            patch(
                "torch._inductor.fx_passes.pad_mm.get_alignment_size"
            ) as mock_get_alignment_size,
            patch("torch._inductor.fx_passes.pad_mm.has_triton") as mock_has_triton,
            patch(
                "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound"
            ) as mock_is_mm_compute_bound,
            patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_should_pad"
            ) as mock_get_cached_should_pad,
            patch("torch._inductor.fx_passes.pad_mm.get_do_bench") as mock_get_do_bench,
            patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_base_mm_benchmark_time"
            ) as mock_get_cached_base_mm_benchmark_time,
            patch(
                "torch._inductor.fx_passes.pad_mm.set_cached_base_mm_benchmark_time"
            ),
            patch("torch._inductor.fx_passes.pad_mm.should_pad") as mock_should_pad,
            patch("torch._inductor.config.force_shape_pad", False),
            patch("torch._inductor.config.deterministic", False),
            patch("torch._inductor.config.post_grad_fusion_options", {}),
        ):
            # Configure mocks
            mock_should_exclude.return_value = False
            mock_get_alignment_size.return_value = 8
            mock_has_triton.return_value = True
            mock_is_mm_compute_bound.return_value = True
            mock_get_cached_should_pad.return_value = None  # No cache hit in old cache
            mock_get_cached_base_mm_benchmark_time.return_value = None

            # Mock benchmarking to return predictable values
            mock_do_bench = MagicMock()
            mock_do_bench.return_value = 1.0
            mock_get_do_bench.return_value = mock_do_bench

            # Make should_pad return True and setup is_padded_faster to return True
            mock_should_pad.return_value = True
            
            # Mock is_padded_faster to return True (pad is faster)
            with patch("torch._inductor.fx_passes.pad_mm.is_padded_faster") as mock_is_padded_faster:
                mock_is_padded_faster.return_value = True

                # Execute: call _should_pad for the first time
                result1 = _should_pad(match, mat1, mat2, torch.ops.aten.mm)

                # Assert: first call executed the function
                self.assertTrue(result1)
                self.assertEqual(mock_do_bench.call_count, 2)  # orig + pad benchmarks

                # Execute: call _should_pad again with same parameters
                mock_do_bench.reset_mock()
                result2 = _should_pad(match, mat1, mat2, torch.ops.aten.mm)

                # Assert: second call retrieved from cache, didn't execute benchmarks
                self.assertTrue(result2)
                self.assertEqual(
                    mock_do_bench.call_count, 0
                )  # No benchmarks on cache hit

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_disk_cache_persistence(self) -> None:
        """Test that _should_pad results persist to disk and can be restored.

        Verifies that when memory cache is cleared (simulating a new process),
        results can still be retrieved from disk cache.
        """
        # Setup: Import the memoized function and memoizer
        from torch._inductor.fx_passes.pad_mm import (
            _should_pad,
            _should_pad_memoizer,
        )
        from unittest.mock import MagicMock

        # Create real tensors (not FakeTensors or MagicMock) to avoid realize_tensor issues
        mat1 = torch.randn(33, 65, dtype=torch.float16, device="cpu")
        mat2 = torch.randn(65, 129, dtype=torch.float16, device="cpu")
        match = MagicMock()

        # Clear the memoizer caches
        _should_pad_memoizer._memoizer._cache = impls._InMemoryCacheImpl()
        _should_pad_memoizer._disk_cache = impls._OnDiskCacheImpl(
            sub_dir=self.sub_dir()
        )

        # Mock dependencies
        with (
            patch(
                "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time"
            ) as mock_should_exclude,
            patch(
                "torch._inductor.fx_passes.pad_mm.get_alignment_size"
            ) as mock_get_alignment_size,
            patch("torch._inductor.fx_passes.pad_mm.has_triton") as mock_has_triton,
            patch(
                "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound"
            ) as mock_is_mm_compute_bound,
            patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_should_pad"
            ) as mock_get_cached_should_pad,
            patch("torch._inductor.fx_passes.pad_mm.get_do_bench") as mock_get_do_bench,
            patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_base_mm_benchmark_time"
            ) as mock_get_cached_base_mm_benchmark_time,
            patch(
                "torch._inductor.fx_passes.pad_mm.set_cached_base_mm_benchmark_time"
            ),
            patch("torch._inductor.fx_passes.pad_mm.should_pad") as mock_should_pad,
            patch("torch._inductor.config.force_shape_pad", False),
            patch("torch._inductor.config.deterministic", False),
            patch("torch._inductor.config.post_grad_fusion_options", {}),
        ):
            # Configure mocks
            mock_should_exclude.return_value = False
            mock_get_alignment_size.return_value = 8
            mock_has_triton.return_value = True
            mock_is_mm_compute_bound.return_value = True
            mock_get_cached_should_pad.return_value = None
            mock_get_cached_base_mm_benchmark_time.return_value = None

            mock_do_bench = MagicMock()
            mock_do_bench.return_value = 1.0
            mock_get_do_bench.return_value = mock_do_bench

            mock_should_pad.return_value = False

            # Execute: first call - caches to both memory and disk
            result1 = _should_pad(match, mat1, mat2, torch.ops.aten.mm)
            self.assertFalse(result1)

            # Clear memory cache to simulate a new process
            _should_pad_memoizer._memoizer._cache = impls._InMemoryCacheImpl()

            # Execute: second call - should retrieve from disk
            mock_do_bench.reset_mock()
            result2 = _should_pad(match, mat1, mat2, torch.ops.aten.mm)

            # Assert: result retrieved from disk, no benchmarking executed
            self.assertFalse(result2)
            self.assertEqual(mock_do_bench.call_count, 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_different_params_different_cache(self) -> None:
        """Test that different parameter combinations result in different cache entries.

        Verifies that the params encoder produces different cache keys for
        different tensor shapes, dtypes, operations, etc.
        """
        # Setup: Import the memoized function
        from torch._inductor.fx_passes.pad_mm import (
            _should_pad,
            _should_pad_memoizer,
        )
        from unittest.mock import MagicMock

        # Clear caches
        _should_pad_memoizer._memoizer._cache = impls._InMemoryCacheImpl()
        _should_pad_memoizer._disk_cache = impls._OnDiskCacheImpl(
            sub_dir=self.sub_dir()
        )

        call_count = 0

        def increment_call_count(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return 1.0

        # Mock dependencies
        with (
            patch(
                "torch._inductor.fx_passes.pad_mm.should_exclude_padding_time"
            ) as mock_should_exclude,
            patch(
                "torch._inductor.fx_passes.pad_mm.get_alignment_size"
            ) as mock_get_alignment_size,
            patch("torch._inductor.fx_passes.pad_mm.has_triton") as mock_has_triton,
            patch(
                "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound"
            ) as mock_is_mm_compute_bound,
            patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_should_pad"
            ) as mock_get_cached_should_pad,
            patch("torch._inductor.fx_passes.pad_mm.get_do_bench") as mock_get_do_bench,
            patch(
                "torch._inductor.fx_passes.pad_mm.get_cached_base_mm_benchmark_time"
            ) as mock_get_cached_base_mm_benchmark_time,
            patch(
                "torch._inductor.fx_passes.pad_mm.set_cached_base_mm_benchmark_time"
            ),
            patch("torch._inductor.fx_passes.pad_mm.should_pad") as mock_should_pad,
            patch("torch._inductor.config.force_shape_pad", False),
            patch("torch._inductor.config.deterministic", False),
            patch("torch._inductor.config.post_grad_fusion_options", {}),
        ):
            # Configure mocks
            mock_should_exclude.return_value = False
            mock_get_alignment_size.return_value = 8
            mock_has_triton.return_value = True
            mock_is_mm_compute_bound.return_value = True
            mock_get_cached_should_pad.return_value = None
            mock_get_cached_base_mm_benchmark_time.return_value = None

            mock_do_bench = MagicMock(side_effect=increment_call_count)
            mock_get_do_bench.return_value = mock_do_bench

            mock_should_pad.return_value = True

            # Create different parameter combinations using real tensors
            match = MagicMock()

            # Combination 1: shape (65, 130) x (130, 257), float32 (needs padding)
            mat1_v1 = torch.randn(65, 130, dtype=torch.float32, device="cpu")
            mat2_v1 = torch.randn(130, 257, dtype=torch.float32, device="cpu")

            # Combination 2: different shape (33, 66) x (66, 129), float32 (needs padding)
            mat1_v2 = torch.randn(33, 66, dtype=torch.float32, device="cpu")
            mat2_v2 = torch.randn(66, 129, dtype=torch.float32, device="cpu")

            # Execute: call with first combination
            call_count = 0
            _should_pad(match, mat1_v1, mat2_v1, torch.ops.aten.mm)
            first_call_count = call_count

            # Execute: call with same combination again (should hit cache)
            call_count = 0
            _should_pad(match, mat1_v1, mat2_v1, torch.ops.aten.mm)
            second_call_count = call_count

            # Execute: call with different combination (should miss cache)
            call_count = 0
            _should_pad(match, mat1_v2, mat2_v2, torch.ops.aten.mm)
            third_call_count = call_count

            # Assert: first call executed benchmarks, second call used cache,
            # third call executed benchmarks again with different params
            self.assertGreater(first_call_count, 0)  # Executed benchmarks
            self.assertEqual(second_call_count, 0)  # Cache hit
            self.assertGreater(third_call_count, 0)  # Different params, cache miss


if __name__ == "__main__":
    run_tests()
