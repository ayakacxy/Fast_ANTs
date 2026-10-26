"""Independent registration with explicit LPS-mm coordinates and optional Metal."""
from ._compat import install as _install_compat
_install_compat()
from .api import __all__

def __getattr__(name):
    from . import api
    value=getattr(api,name)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals())|set(__all__))
