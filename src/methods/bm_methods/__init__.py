"""
Benchmark methods for feature selection comparison.

This module contains baseline feature selection methods that are commonly
used in scRNA-seq analysis. All methods follow the same interface as the
GJSD pipeline for fair comparison.

Package-based methods (use actual implementations):
- SeuratHVGMethod: Seurat v3 HVG via scanpy
- ScanpyHVGMethod: Scanpy native HVG
- WilcoxonMethod: Wilcoxon test via scanpy
- TTestMethod: Student's t-test
- DevianceMethod: scry package (with Python fallback)
- TrikuMethod: Triku package
- COSGMethod: COSG package
- M3DropMethod: M3Drop-style dropout analysis

Simple baseline methods:
- MutualInfoMethod: Mutual information (sklearn)
- FoldChangeMethod: Simple fold-change
"""

from .base_method import BaseBenchmarkMethod

# Package-based methods (use actual implementations)
from .seurat_hvg_method import SeuratHVGMethod
from .scanpy_hvg_method import ScanpyHVGMethod
from .wilcoxon_method import WilcoxonMethod
from .ttest_method import TTestMethod
from .deviance_method import DevianceMethod

# Additional package methods (require installation)
from .triku_method import TrikuMethod
from .cosg_method import COSGMethod
from .m3drop_method import M3DropMethod

# Simple baseline methods
from .mutual_info_method import MutualInfoMethod
from .fold_change_method import FoldChangeMethod

# Method registry for easy access
METHOD_REGISTRY = {
    # Standard package-based methods (always available)
    'seurat_hvg': SeuratHVGMethod,
    'scanpy_hvg': ScanpyHVGMethod,
    'wilcoxon': WilcoxonMethod,
    'ttest': TTestMethod,
    'deviance': DevianceMethod,
    
    # Additional package methods (require separate installation)
    'triku': TrikuMethod,
    'cosg': COSGMethod,
    'm3drop': M3DropMethod,
    
    # Simple baselines
    'mutual_info': MutualInfoMethod,
    'fold_change': FoldChangeMethod,
}


def get_method(method_name: str, **kwargs):
    """
    Factory function to get a benchmark method by name.
    
    Args:
        method_name: Name of the method (see METHOD_REGISTRY keys)
        **kwargs: Arguments to pass to the method constructor
    
    Returns:
        Instance of the requested benchmark method
    
    Example:
        >>> method = get_method('wilcoxon', 
        ...                     adata=adata,
        ...                     tf_list=tf_list,
        ...                     target_cell_type='T cells',
        ...                     background_cell_type='other')
        >>> results = method.run()
    
    Available methods:
        Standard (no extra packages needed):
        - seurat_hvg: Seurat v3 HVG (via scanpy)
        - scanpy_hvg: Scanpy native HVG
        - wilcoxon: Wilcoxon rank-sum test (via scanpy)
        - ttest: Student's t-test
        - deviance: Deviance-based (scry package, with fallback)
        - mutual_info: Mutual information (sklearn)
        - fold_change: Simple fold change
        
        Require additional packages:
        - triku: pip install triku
        - cosg: pip install COSG
        - m3drop: Python implementation (no extra package)
    """
    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method: {method_name}. "
            f"Available methods: {list(METHOD_REGISTRY.keys())}"
        )
    
    method_class = METHOD_REGISTRY[method_name]
    return method_class(**kwargs)


def list_methods():
    """
    List all available benchmark methods.
    
    Returns:
        List of method names
    """
    return list(METHOD_REGISTRY.keys())


def list_standard_methods():
    """
    List methods that don't require additional package installation.
    
    Returns:
        List of method names
    """
    return ['seurat_hvg', 'scanpy_hvg', 'wilcoxon', 'ttest', 
            'deviance', 'mutual_info', 'fold_change']


def list_package_methods():
    """
    List methods that require additional package installation.
    
    Returns:
        Dict mapping method name to package requirement
    """
    return {
        'triku': 'pip install triku',
        'cosg': 'pip install COSG',
        'm3drop': 'No extra package (Python implementation)',
    }


__all__ = [
    'BaseBenchmarkMethod',
    # Standard methods
    'SeuratHVGMethod',
    'ScanpyHVGMethod',
    'WilcoxonMethod',
    'TTestMethod',
    'DevianceMethod',
    'MutualInfoMethod',
    'FoldChangeMethod',
    # Package methods
    'TrikuMethod',
    'COSGMethod',
    'M3DropMethod',
    # Utilities
    'get_method',
    'list_methods',
    'list_standard_methods',
    'list_package_methods',
    'METHOD_REGISTRY',
]
