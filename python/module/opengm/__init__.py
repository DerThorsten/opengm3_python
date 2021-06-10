from ._opengm import *


graphical_model_types = {
    "UniformSpaceGraphicalModel": UniformSpaceGraphicalModel,
    "BinarySpaceGraphicalModel": BinarySpaceGraphicalModel,
}

# thismodule = sys.modules[__name__]

minimizer_names = [
    "BeliefPropergation",
    "Icm",
    "FactorIcm",
    "SelfFusion",
    "ChainedMinimizers",
]


FuseGm = BinarySpaceGraphicalModel


def meta_factory(factory_cls):
    class MetaFactory(factory_cls):
        def __init__(self, settings=None, **kwargs):
            if settings is None:
                settings = factory_cls.Settings()
                for k, v in kwargs.items():
                    setattr(settings, k, v)
            elif not isinstance(settings, factory_cls.Settings):
                raise RuntimeError(f"settings must be of type f{factory_cls.Settings}")
            super().__init__(settings=settings)

    return MetaFactory


def extend_callbacks():
    callback_names = ["VerboseCallback"]
    for callback_name in callback_names:
        for graphical_model_name, graphical_model_type in graphical_model_types.items():
            callback_cls = getattr(_opengm, f"{callback_name}{graphical_model_name}")
            setattr(graphical_model_type, callback_name, callback_cls)


extend_callbacks()
del extend_callbacks


def extend_minimizers():
    for graphical_model_name, graphical_model_type in graphical_model_types.items():

        label_fuser_cls = getattr(_opengm, f"LabelFuser{graphical_model_name}")
        label_fuser_settings_cls = getattr(
            _opengm, f"LabelFuserSettings{graphical_model_name}"
        )

        setattr(label_fuser_cls, f"Settings", label_fuser_settings_cls)
        setattr(graphical_model_type, f"LabelFuser", label_fuser_cls)

        for minimizer_name in minimizer_names:

            minimizer_cls = getattr(_opengm, f"{minimizer_name}{graphical_model_name}")
            factory_cls = getattr(
                _opengm, f"{minimizer_name}Factory{graphical_model_name}"
            )
            meta_factory_cls = meta_factory(factory_cls)
            settings_cls = getattr(
                _opengm, f"{minimizer_name}Settings{graphical_model_name}"
            )

            setattr(minimizer_cls, f"Factory", meta_factory_cls)
            setattr(minimizer_cls, f"Settings", settings_cls)

            setattr(settings_cls, f"Factory", meta_factory_cls)
            setattr(settings_cls, f"Minimizer", minimizer_cls)

            setattr(factory_cls, f"Minimizer", minimizer_cls)
            setattr(factory_cls, f"Settings", settings_cls)

            setattr(graphical_model_type, f"{minimizer_name}", minimizer_cls)


extend_minimizers()
del extend_minimizers


def pure_python():
    """
    hello
    """
    pass
