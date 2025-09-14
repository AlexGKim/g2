# Import sources first to ensure they're available
try:
    from . import sources
    from .core import *
except ImportError as e:
    # For documentation generation, provide minimal imports
    import warnings
    warnings.warn(f"Failed to import g2 modules: {e}. Using minimal imports for documentation.")
    pass