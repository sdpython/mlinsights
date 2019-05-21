"""
@file
@brief Caches to cache training.
"""
import numpy

_caches = {}


class MLCache:
    """
    Implements a cache to reduce the number of trainings
    a grid search has to do.
    """

    def __init__(self, name):
        """
        @param      name        name of the cache
        """
        self.name = name
        self.cached = {}

    def cache(self, params, value):
        """
        Caches one object.

        @param      params  dictionary of parameters
        @param      value   value to cache
        """
        key = MLCache.as_key(params)
        if key in self.cached:
            raise KeyError("Key {0} already exists".format(params))
        self.cached[key] = value

    def get(self, params, default=None):
        """
        Retrieves an element from the cache.

        @param      params  dictionary of parameters
        @param      default if not found
        @return             value or None if it does not exists
        """
        key = MLCache.as_key(params)
        return self.cached.get(key, default)

    @staticmethod
    def as_key(params):
        """
        Converts a list of parameters into a key.

        @param      params      dictionary
        @return                 key as a string
        """
        els = []
        for k, v in sorted(params.items()):
            if isinstance(v, (int, float, str)):
                sv = str(v)
            elif isinstance(v, numpy.ndarray):
                sv = str(id(v))
            elif v is None:
                sv = ""
            else:
                raise TypeError(
                    "Unable to create a key with value '{0}':{1}".format(k, v))
            els.append((k, sv))
        return str(els)

    def __len__(self):
        """
        Returns the number of cached items.
        """
        return len(self.cached)

    def items(self):
        """
        Enumerates all cached items.
        """
        for item in self.cached.items():
            yield item

    def keys(self):
        """
        Enumerates all cached keys.
        """
        for k in self.cached.keys():  # pylint: disable=C0201
            yield k

    @staticmethod
    def create_cache(name):
        """
        Creates a new cache.

        @param      name        name
        @return                 created cache
        """
        global _caches  # pylint: disable=W0603
        if name in _caches:
            raise RuntimeError("cache '{0}' already exists.".format(name))

        cache = MLCache(name)
        _caches[name] = cache
        return cache

    @staticmethod
    def remove_cache(name):
        """
        Removes a cache with a given name.

        @param      name        name
        """
        global _caches  # pylint: disable=W0603
        del _caches[name]

    @staticmethod
    def get_cache(name):
        """
        Gets a cache with a given name.

        @param      name        name
        @return                 created cache
        """
        global _caches  # pylint: disable=W0603
        return _caches[name]

    @staticmethod
    def has_cache(name):
        """
        Tells if cache *name* is present.

        @param      name        name
        @return                 boolean
        """
        global _caches  # pylint: disable=W0603
        return name in _caches
