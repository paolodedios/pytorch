# Owner(s): ["module: inductor"]
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor
from os import environ
from random import randint
from typing_extensions import Self

from torch._inductor import pcache
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


key_gen: Generator[pcache.Key, None, None] = (
    f"dummy_key_{randint(0, 100000)}" for _ in iter(int, 1)
)
value_gen: Generator[pcache.Value, None, None] = (
    f"dummy_value_{randint(0, 100000)}".encode() for _ in iter(int, 1)
)


Caches: list[type[pcache.Cache]] = [pcache.InMemoryCache]
AsyncCaches: list[type[pcache.AsyncCache]] = [pcache.InductorOnDiskCache]


@instantiate_parametrized_tests
class CacheTest(TestCase):
    @parametrize("Cache", Caches)
    def test_get_set_get(self: Self, Cache: type[pcache.Cache]) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        cache: pcache.Cache = Cache()

        # make sure our key is fresh
        while cache.get(key) is not None:
            key = next(key_gen)

        # first get should return None, no hit
        self.assertIsNone(cache.get(key))
        # put should return True, having set key -> value
        self.assertTrue(cache.put(key, value))
        # second get should return value, hit
        self.assertEqual(cache.get(key), value)

    @parametrize("Cache", Caches)
    def test_set_set(self: Self, Cache: type[pcache.Cache]) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        cache: pcache.Cache = Cache()

        if cache.get(key) is None:
            # if key isn't already cached, cache it
            self.assertTrue(cache.put(key, value))

        # second put should not update the value
        self.assertFalse(cache.put(key, value))

    def test_in_memory_cache_from_env_var(
        self: Self, Cache: type[pcache.InMemoryCache] = pcache.InMemoryCache
    ) -> None:
        key_1: pcache.Key = next(key_gen)
        value_1: pcache.Value = next(value_gen)

        key_2: pcache.Key = next(key_gen)
        while key_2 == key_1:
            key_2 = next(key_gen)
        value_2: pcache.Value = next(value_gen)

        key_3: pcache.Key = next(key_gen)
        while key_3 in (key_1, key_2):
            key_3 = next(key_gen)

        env_var = "INMEMORYCACHE_TEST"
        env_val = f"{key_1},{value_1!r};{key_2},{value_2!r}"
        environ[env_var] = env_val

        cache = Cache.from_env_var(env_var)

        # key_1 -> value_1 is in env_val, so we should hit
        self.assertEqual(cache.get(key_1), value_1)
        # key_2 -> value_2 is in env_val, so we should hit
        self.assertEqual(cache.get(key_2), value_2)
        # key_3 -> value_3 is not in env_val, so we should miss
        self.assertIsNone(cache.get(key_3))


@instantiate_parametrized_tests
class AsyncCacheTest(TestCase):
    @parametrize("AsyncCache", AsyncCaches)
    @parametrize("Executor", [ThreadPoolExecutor, None])
    def test_get_set_get(
        self: Self,
        AsyncCache: type[pcache.AsyncCache],
        Executor: type[ThreadPoolExecutor] | None = None,
    ) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        async_cache: pcache.AsyncCache = AsyncCache()
        executor: ThreadPoolExecutor = Executor() if Executor is not None else None

        if executor is None:
            # make sure our key is fresh
            while async_cache.get(key, executor=None) is not None:
                key = next(key_gen)

            # first get should miss
            self.assertIsNone(async_cache.get(key, executor=None))
            # put should set key -> value mapping
            self.assertTrue(async_cache.put(key, value, executor=None))
            # second get should hit
            self.assertEqual(async_cache.get(key, executor=None), value)
        else:
            # make sure our key is fresh
            while async_cache.get(key, executor=executor).result() is not None:
                key = next(key_gen)

            # first get should miss
            self.assertIsNone(async_cache.get(key, executor=executor).result())
            # put should set key -> value mapping
            self.assertTrue(async_cache.put(key, value, executor=executor).result())
            # second get should hit
            self.assertEqual(async_cache.get(key, executor=executor).result(), value)
            executor.shutdown()

    @parametrize("AsyncCache", AsyncCaches)
    @parametrize("Executor", [ThreadPoolExecutor, None])
    def test_set_set(
        self: Self,
        AsyncCache: type[pcache.AsyncCache],
        Executor: type[ThreadPoolExecutor] | None = None,
    ) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        async_cache: pcache.AsyncCache = AsyncCache()
        executor: ThreadPoolExecutor = Executor() if Executor is not None else None

        if executor is None:
            if async_cache.get(key, executor=None) is None:
                # set key -> value mapping if unset
                self.assertTrue(async_cache.put(key, value, executor=None))
            # second put should not override the prior put
            self.assertFalse(async_cache.put(key, value, executor=None))
        else:
            if async_cache.get(key, executor=executor).result() is None:
                # set key -> value mapping if unset
                self.assertTrue(async_cache.put(key, value, executor=executor).result())
            # second put should not override the prior put
            self.assertFalse(async_cache.put(key, value, executor=executor).result())
            executor.shutdown()

    @parametrize("AsyncCache", AsyncCaches)
    def test_set_set_one_pass(
        self: Self,
        AsyncCache: type[pcache.AsyncCache],
        Executor: type[ThreadPoolExecutor] = ThreadPoolExecutor,
    ) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        async_cache: pcache.AsyncCache = AsyncCache()
        executor: ThreadPoolExecutor = Executor()

        # make sure our key is fresh
        while async_cache.get(key, executor=executor).result() is not None:
            key = next(key_gen)

        put_1: Future[bool] = async_cache.put(key, value, executor=executor)
        put_2: Future[bool] = async_cache.put(key, value, executor=executor)

        # only one put should succeed
        self.assertTrue(put_1.result() ^ put_2.result())
        executor.shutdown()


if __name__ == "__main__":
    run_tests()
