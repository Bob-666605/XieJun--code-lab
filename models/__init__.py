from .base import BaseDenoiser
from .traditional import TraditionalDenoiser

__all__ = ['BaseDenoiser', 'TraditionalDenoiser']

try:
    from .mash import MASHDenoiser
    from .dncnn import DnCNN
    __all__ += ['MASHDenoiser', 'DnCNN']
except ImportError:
    pass
