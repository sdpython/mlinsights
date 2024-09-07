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
        self.count_ = {}

    def cache(self, params, value):
        """
        Caches one object.

        @param      params  dictionary of parameters
        @param      value   value to cache
        """
        key = MLCache.as_key(params)
        assert key not in self.cached, f"Key {params} already exists"
        self.cached[key] = value
        self.count_[key] = 0

    def get(self, params, default=None):
        """
        Retrieves an element from the cache.

        @param      params  dictionary of parameters
        @param      default if not found
        @return             value or None if it does not exists
        """
        key = MLCache.as_key(params)
        res = self.cached.get(key, default)
        if res != default:
            self.count_[key] += 1
        return res

    def count(self, params):
        """
        Retrieves the number of times
        an elements was retrieved from the cache.

        @param      params  dictionary of parameters
        @return             int
        """
        key = MLCache.as_key(params)
        return self.count_.get(key, 0)

    @staticmethod
    def as_key(params):
        """
        Converts a list of parameters into a key.

        @param      params      dictionary
        @return                 key as a string
        """
        if isinstance(params, str):
            return params
        els = []
        for k, v in sorted(params.items()):
            if isinstance(v, (int, float, str)):
                sv = str(v)
            elif isinstance(v, tuple):
                assert all(
                    isinstance(e, (int, float, str)) for e in v
                ), f"Unable to create a key with value {k!r}:{v}"
                return str(v)
            elif isinstance(v, numpy.ndarray):
                # id(v) may have been better but
                # it does not play well with joblib.
                sv = hash(v.tobytes())
            elif v is None:
                sv = ""
            else:
                raise TypeError(f"Unable to create a key with value {k!r}:{v!r}")
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
        yield from self.cached.items()

    def keys(self):
        """
        Enumerates all cached keys.
        """
        yield from self.cached.keys()

    @staticmethod
    def create_cache(name):
        """
        Creates a new cache.

        @param      name        name
        @return                 created cache
        """
        global _caches
        assert name not in _caches, f"cache {name!r} already exists."

        cache = MLCache(name)
        _caches[name] = cache
        return cache

    @staticmethod
    def remove_cache(name):
        """
        Removes a cache with a given name.

        @param      name        name
        """
        global _caches
        del _caches[name]

    @staticmethod
    def get_cache(name):
        """
        Gets a cache with a given name.

        @param      name        name
        @return                 created cache
        """
        global _caches
        return _caches[name]

    @staticmethod
    def has_cache(name):
        """
        Tells if cache *name* is present.

        @param      name        name
        @return                 boolean
        """
        global _caches
        return name in _caches
