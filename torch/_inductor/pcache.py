from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property
from os import getenv
from pathlib import Path
from typing_extensions import Self


# Key represents data used to query from cache instances
Key = str
# Value represents data stored in/retrieved from cache instances
Value = bytes


class Cache:
    """
    Abstract base class for cache implementations.
    Provides the interface for basic get and put methods for storing and retrieving data.
    """

    def get(self: Self, key: Key) -> Value | None:
        """
        Retrieve the value associated with the given key from the cache.
        Args:
            key: The key used to query the cache.
        Returns:
            The value associated with the key, or None if not found.
        """
        raise NotImplementedError

    def put(self: Self, key: Key, value: Value) -> bool:
        """
        Store the given value in the cache with the associated key.
        Args:
            key: The key to associate with the value.
            value: The value to be stored in the cache.
        Returns:
            True if the value was stored successfully, False otherwise.
        """
        raise NotImplementedError


class InMemoryCache(Cache):
    """
    A simple in-memory cache implementation.
    Stores cache data in a dictionary for fast lookups and insertions.
    """

    def __init__(self: Self) -> None:
        """
        Initialize the in-memory cache.
        """
        self._cache: dict[Key, Value] = {}

    def get(self: Self, key: Key) -> Value | None:
        """
        Retrieve the value associated with the given key from the cache.
        Args:
            key: The key used to query the cache.
        Returns:
            The value associated with the key, or None if not found.
        """
        return self._cache.get(key, None)

    def put(self: Self, key: Key, value: Value) -> bool:
        """
        Store the given value in the cache with the associated key.
        Args:
            key: The key to associate with the value.
            value: The value to be stored in the cache.
        Returns:
            True if the value was stored successfully, False otherwise.
        """
        if key in self._cache:
            return False
        else:
            self._cache[key] = value
            return True

    @classmethod
    def from_env_var(cls, env_var: str) -> Self:
        """
        Create a new in-memory cache instance from an environment variable.
        The environment variable should contain key-value pairs separated by ';',
        with each pair formatted as 'key,value'. The value should be a string
        representation of bytes (e.g., b'...').
        Args:
            env_var: The environment variable containing cache data.
        Returns:
            A new in-memory cache instance populated with data from the environment variable.
        Raises:
            ValueError: If a key is associated with two distinct values.
        """
        cache: Self = cls()
        env_val: str | None = getenv(env_var, None)

        if env_val is not None:
            for kv_pair in env_val.split(";"):
                key, value = kv_pair.split(",")
                # value is a str repr of bytes, remove b' prefix and ' suffix
                value = value[2:-1].encode()
                # duplicates are ok, as long as key and value are both equal
                # if key has two distinct values, fail as this is likely a user error
                if (not cache.put(key, value)) and (cache.get(key) != value):
                    raise ValueError(
                        f"Duplicated values for key {key}, got {cache.get(key)!r} and {value!r}!"
                    )

        return cache


class AsyncCache:
    """
    Abstract base class for asynchronous cache implementations.
    Provides get and put methods with support for asynchronous execution.
    """

    def _sync_get(self: Self, key: Key) -> Value | None:
        """
        Retrieve the value associated with the given key from the cache synchronously.
        Args:
            key: The key used to query the cache.
        Returns:
            The value associated with the key, or None if not found.
        """
        raise NotImplementedError

    def get(
        self: Self, key: Key, executor: ThreadPoolExecutor | None = None
    ) -> Future[Value | None] | Value | None:
        """
        Retrieve the value associated with the given key from the cache.
        Args:
            key: The key used to query the cache.
            executor: The executor to use for asynchronous execution, or None for synchronous execution.
        Returns:
            The value associated with the key, or None if not found. If an executor is provided,
            returns a Future representing the result of the asynchronous operation.
        """
        if executor is None:
            return self._sync_get(key)
        else:
            return executor.submit(self._sync_get, key)

    def _sync_put(self: Self, key: Key, value: Value) -> bool:
        """
        Store the given value in the cache with the associated key synchronously.
        Args:
            key: The key to associate with the value.
            value: The value to be stored in the cache.
        Returns:
            True if the value was stored successfully, False otherwise.
        """
        raise NotImplementedError

    def put(
        self: Self, key: Key, value: Value, executor: ThreadPoolExecutor | None = None
    ) -> Future[bool] | bool:
        """
        Store the given value in the cache with the associated key.
        Args:
            key: The key to associate with the value.
            value: The value to be stored in the cache.
            executor: The executor to use for asynchronous execution, or None for synchronous execution.
        Returns:
            True if the value was stored successfully, False otherwise. If an executor is provided,
            returns a Future representing the result of the asynchronous operation.
        """
        if executor is None:
            return self._sync_put(key, value)
        else:
            return executor.submit(self._sync_put, key, value)


class OnDiskCache(AsyncCache):
    """
    Abstract base class for on-disk cache implementations.
    Provides get and put methods for storing and retrieving data on disk,
    with support for asynchronous execution.
    """

    @property
    def base_dir(self: Self) -> Path:
        """
        Get the base directory for the on-disk cache.
        Returns:
            The base directory for the on-disk cache.
        """
        raise NotImplementedError

    def _fpath_from_key(self: Self, key: Key) -> Path:
        """
        Get the file path associated with the given key.
        Args:
            key: The key used to query the cache.
        Returns:
            The file path associated with the key.
        """
        return self.base_dir / key

    def _sync_get(self: Self, key: Key) -> Value | None:
        """
        Retrieve the value associated with the given key from the cache synchronously.
        Args:
            key: The key used to query the cache.
        Returns:
            The value associated with the key, or None if not found.
        """
        fpath = self._fpath_from_key(key)
        return fpath.read_bytes() if fpath.is_file() else None

    def _sync_put(self: Self, key: Key, value: Value) -> bool:
        """
        Store the given value in the cache with the associated key synchronously.
        Args:
            key: The key to associate with the value.
            value: The value to be stored in the cache.
        Returns:
            True if the value was stored successfully, False otherwise.
        """
        fpath = self._fpath_from_key(key)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            # "x" mode is exclusive creation, meaning the file will be created
            # iff the file does not already exist (atomic w/o overwrite)
            with open(fpath, "xb") as fp:
                fp.write(value)
        except FileExistsError:
            return False
        return True


class InductorOnDiskCache(OnDiskCache):
    """
    A specific on-disk cache implementation for Inductor.
    Uses the default cache directory provided by Inductor.
    """

    @cached_property
    def base_dir(self: Self) -> Path:
        """
        Get the base directory for the on-disk cache.
        Returns:
            The base directory for the on-disk cache.
        """
        from torch._inductor.runtime.runtime_utils import default_cache_dir

        return Path(default_cache_dir(), "pcache")
