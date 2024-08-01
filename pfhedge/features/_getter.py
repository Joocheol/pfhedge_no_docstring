from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Tuple
from typing import Type
from typing import Union

from ._base import Feature


class FeatureFactory:

    _features: Dict[str, Type[Feature]]

    # singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls._instance._features = OrderedDict()
        return cls._instance

    def register_feature(self, name: str, cls: Type[Feature]) -> None:
        
        self._features[name] = cls

    def named_features(self) -> Iterator[Tuple[str, Type[Feature]]]:
        
        for name, feature in self._features.items():
            if feature is not None:
                yield name, feature

    def names(self) -> Iterator[str]:
        
        for name, _ in self.named_features():
            yield name

    def features(self) -> Iterator[Type[Feature]]:
        
        for _, feature in self.named_features():
            yield feature

    def get_class(self, name: str) -> Type[Feature]:
        
        if name not in self.names():
            raise KeyError(
                f"{name} is not a valid name. "
                "Use pfhedge.features.list_feature_names() to see available names."
            )
        return self._features[name]

    def get_instance(self, name: str, **kwargs: Any) -> Feature:
        
        return self.get_class(name)(**kwargs)  # type: ignore


def get_feature(feature: Union[str, Feature], **kwargs: Any) -> Feature:
    
    if isinstance(feature, str):
        feature = FeatureFactory().get_instance(feature, **kwargs)
    elif not isinstance(feature, Feature):
        raise TypeError(f"{feature} is not an instance of Feature.")
    return feature


def list_feature_dict() -> dict:
    return dict(FeatureFactory().named_features())


def list_feature_names() -> list:
    
    return sorted(list(FeatureFactory().names()))


def list_features() -> list:
    return list(FeatureFactory().features())
