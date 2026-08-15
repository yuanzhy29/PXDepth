"""Small component registries used by PXDepth configuration builders.

The release intentionally keeps this mechanism minimal.  A registry maps a
short string from a JSON config to a Python class or function.  Third-party
projects can register their own implementation from a module listed in the
config's optional ``imports`` field, without modifying PXDepth source.
"""

from importlib import import_module
from typing import Any, Callable, Dict, Optional, TypeVar


T = TypeVar("T")


class Registry:
    """Map configurable names to callables and instantiate them from dictionaries.

    Args:
        name: Human-readable component category used in error messages.

    The registry accepts both short registered names and dotted Python paths.
    A dotted path such as ``my_package.models.CustomEncoder`` is imported lazily,
    which makes small out-of-tree experiments possible without editing this repo.
    """

    def __init__(self, name: str) -> None:
        """Create an empty registry for one component category.

        Args:
            name: Human-readable category used in validation error messages.

        Returns:
            ``None``. Registered items are stored in a new private mapping.
        """
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(
        self,
        value: Optional[T] = None,
        name: Optional[str] = None,
    ) -> Callable[[T], T] | T:
        """Register a class or function, directly or as a decorator.

        Args:
            value: Callable to register.  Omit it when using decorator syntax.
            name: Optional config name.  The callable's ``__name__`` is used when
                no explicit name is supplied.

        Returns:
            The original callable, allowing ``@REGISTRY.register()`` usage.
        """

        def add(item: T) -> T:
            """Insert one callable and return it unchanged.

            Args:
                item: Class or function supplied directly or by decorator use.

            Returns:
                The original object, preserving normal decorator semantics.
            """
            key = name or getattr(item, "__name__", None)
            if not key:
                raise ValueError(f"A {self.name} registration needs an explicit name.")
            if key in self._items and self._items[key] is not item:
                raise KeyError(f"{self.name} '{key}' is already registered.")
            self._items[key] = item  # type: ignore[assignment]
            return item

        return add if value is None else add(value)

    def get(self, name: str) -> Callable[..., Any]:
        """Resolve a registered name or dotted import path to a callable.

        Args:
            name: Registered short name or ``package.module.callable`` path.

        Returns:
            Resolved class or function.
        """
        if name in self._items:
            return self._items[name]
        if "." in name:
            module_name, attribute = name.rsplit(".", 1)
            value = getattr(import_module(module_name), attribute)
            if not callable(value):
                raise TypeError(f"Resolved {self.name} '{name}' is not callable.")
            return value
        available = ", ".join(sorted(self._items)) or "none"
        raise KeyError(f"Unknown {self.name} '{name}'. Available: {available}")

    def build(self, config: Dict[str, Any], **defaults: Any) -> Any:
        """Instantiate one component from a ``type`` plus constructor arguments.

        Args:
            config: Dictionary containing a required ``type`` key. Remaining
                entries are passed to the resolved callable as keyword arguments.
            **defaults: Values used only when the config does not define the key.

        Returns:
            Constructed component instance.
        """
        if not isinstance(config, dict):
            raise TypeError(f"{self.name} config must be a dictionary.")
        params = dict(defaults)
        params.update(config)
        type_name = params.pop("type", None)
        if not isinstance(type_name, str) or not type_name:
            raise ValueError(f"{self.name} config requires a non-empty 'type'.")
        return self.get(type_name)(**params)

    def names(self) -> tuple[str, ...]:
        """List the short names currently registered in this category.

        Returns:
            Lexicographically sorted tuple of names. Dotted paths are resolved
            lazily and therefore do not appear unless explicitly registered.
        """
        return tuple(sorted(self._items))


MODELS = Registry("model")
ENCODERS = Registry("encoder")
PREDICTORS = Registry("predictor")
