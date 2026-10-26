"""CPU ordinary FP32 operators and stage implementations."""
def __getattr__(name):
    from importlib import import_module
    module=import_module(__name__+".kernels")
    try:return getattr(module,name)
    except AttributeError:raise AttributeError(name) from None
