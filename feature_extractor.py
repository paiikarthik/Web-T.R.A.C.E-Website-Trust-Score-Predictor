"""
Web T.R.A.C.E — Feature Extractor Compatibility Layer.
Re-exports the unified feature extraction engine from features.py.
"""

from features import (
    GENERAL_FEATURE_NAMES as FEATURE_COLUMNS,
    extract_features,
    extract_features_df,
    extract_host_clean,
    get_domain_group,
    parse_url_components,
    shannon_entropy,
)

__all__ = [
    "FEATURE_COLUMNS",
    "extract_features",
    "extract_features_df",
    "extract_host_clean",
    "get_domain_group",
    "parse_url_components",
    "shannon_entropy",
]
