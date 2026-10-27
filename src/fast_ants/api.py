"""Stable, lazily imported public Python API."""
from importlib import import_module

_EXPORTS = {'Grid': ('fast_ants.core.types', 'Grid'),
 'Image': ('fast_ants.core.types', 'Image'),
 'RegistrationResult': ('fast_ants.core.types', 'RegistrationResult'),
 'AffineTransform': ('fast_ants.core.transforms', 'AffineTransform'),
 'CompositeTransform': ('fast_ants.core.transforms', 'CompositeTransform'),
 'load_image': ('fast_ants.io.images', 'load_image'),
 'load_images': ('fast_ants.io.images', 'load_images'),
 'save_image': ('fast_ants.io.images', 'save_image'),
 'DisplacementField': ('fast_ants.core.fields', 'DisplacementField'),
 'apply_image': ('fast_ants.backends.cpu.fp64.sampling', 'apply_image'),
 'sample_image': ('fast_ants.backends.cpu.fp64.sampling', 'sample_image'),
 'transform_to_field': ('fast_ants.backends.cpu.fp64.sampling', 'transform_to_field'),
 'load_affine': ('fast_ants.io.transforms', 'load_affine'),
 'save_affine': ('fast_ants.io.transforms', 'save_affine'),
 'load_field': ('fast_ants.io.transforms', 'load_field'),
 'save_field': ('fast_ants.io.transforms', 'save_field'),
 'load_ants_transform_list': ('fast_ants.io.transforms', 'load_ants_transform_list'),
 'LinearStage': ('fast_ants.core.config', 'LinearStage'),
 'register_linear': ('fast_ants.backends.cpu.fp64.linear', 'register_linear'),
 'SyNStage': ('fast_ants.core.config', 'SyNStage'),
 'register': ('fast_ants.backends.cpu.fp64.registration', 'register'),
 'register32': ('fast_ants.backends.cpu.fp32.registration', 'register32'),
 'register_metal': ('fast_ants.backends.metal.registration', 'register_metal'),
 'register_cuda': ('fast_ants.backends.cuda.registration', 'register_cuda')}
__all__ = ['Grid',
 'Image',
 'RegistrationResult',
 'AffineTransform',
 'CompositeTransform',
 'load_image',
 'load_images',
 'save_image',
 'DisplacementField',
 'apply_image',
 'sample_image',
 'transform_to_field',
 'load_affine',
 'save_affine',
 'load_field',
 'save_field',
 'load_ants_transform_list',
 'LinearStage',
 'register_linear',
 'SyNStage',
 'register',
 'register32',
 'register_metal',
 'register_cuda']

def register_metal(*args, **kwargs):
    """Load the optional GPU implementation only when registration is called."""
    from .backends.metal.registration import register_metal as implementation
    return implementation(*args, **kwargs)

def register_cuda(*args, **kwargs):
    """Load the optional CUDA implementation only when registration is called."""
    from .backends.cuda.registration import register_cuda as implementation
    return implementation(*args, **kwargs)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,symbol=_EXPORTS[name]
    value=getattr(import_module(module),symbol)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals())|set(__all__))
