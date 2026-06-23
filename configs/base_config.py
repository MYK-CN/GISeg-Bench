"""
GISeg-Bench  Base Configuration Class
=======================================
Lightweight ``Config`` that supports both dict-style and dot-access.

Usage::

    cfg = Config({"lr": 1e-4, "epochs": 100})
    print(cfg.lr)        # 1e-4
    print(cfg["epochs"]) # 100

    # Nested access
    cfg.merge({"optimizer": "adamw", "batch_size": 4})
    print(cfg.optimizer) # adamw

    # Export
    d = cfg.to_dict()    # plain dict
"""


class Config:
    """Unified config object — attribute + item access.

    All the project subsystems (trainer, inference, metrics) accept
    either a dict or a ``Config`` instance.
    """

    def __init__(self, data=None, **kwargs):
        self.__dict__["_data"] = {}
        if data is not None:
            if isinstance(data, Config):
                data = data._data
            elif isinstance(data, dict):
                pass
            else:
                data = vars(data) if hasattr(data, "__dict__") else {}
            self._data.update(data)
        self._data.update(kwargs)

    # ------------------------------------------------------------------
    #  Attribute access
    # ------------------------------------------------------------------
    def __getattr__(self, key):
        if key.startswith("_"):
            return object.__getattribute__(self, key)
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value

    # ------------------------------------------------------------------
    #  Dict-like access
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    # ------------------------------------------------------------------
    #  Merge
    # ------------------------------------------------------------------
    def merge(self, other):
        """Update in-place from another dict / Config / namespace.

        Existing keys are overwritten.
        """
        if isinstance(other, Config):
            other = other._data
        elif not isinstance(other, dict):
            other = vars(other) if hasattr(other, "__dict__") else {}
        self._data.update(other)
        return self

    def copy(self):
        """Return a shallow copy."""
        return Config(self._data.copy())

    # ------------------------------------------------------------------
    #  Export
    # ------------------------------------------------------------------
    def to_dict(self):
        """Return a plain dict copy."""
        return dict(self._data)

    # ------------------------------------------------------------------
    #  Representation
    # ------------------------------------------------------------------
    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"Config({items})"

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)
