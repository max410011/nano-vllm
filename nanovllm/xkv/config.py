"""
Configuration classes for xKV compression.

This module provides configuration for cross-layer KV-Cache compression:
- LayerGroup: Defines a group of layers to be merged together
- xKVConfig: Main configuration class for xKV compression settings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LayerGroup:
    """
    Represents a group of layers to be merged.
    
    For SVD method: rank_k and rank_v control the compression rank
    For SLERP method: slerp_t and slerp_gamma control interpolation
    """
    layers: List[int] = field(default_factory=list)

    # SVD parameters (if 'layer_merge_impl' == "svd")
    rank_k: Optional[int] = None
    rank_v: Optional[int] = None

    # SLERP parameters (if 'layer_merge_impl' == "slerp")
    slerp_t: Optional[float] = None
    slerp_gamma: Optional[float] = None

    def __post_init__(self):
        if not self.layers:
            raise ValueError("LayerGroup must have at least one layer index.")


@dataclass
class xKVConfig:
    """
    Configuration for xKV compression.

    Supports two compression methods:
    - 'svd': Uses SVD decomposition with rank truncation
    - 'slerp': Uses SLERP interpolation (only for group_size=2)

    Supports sparse attention (ShadowKV-style) for further memory savings:
    - enable_sparse: Enable chunk-based sparse KV selection
    - chunk_size: Tokens per chunk for landmark computation
    - sparse_budget: Number of chunks to select during decode
    - num_outliers: Number of outlier chunks to keep as static cache
    """
    num_layers: Optional[int] = None
    layer_merge_impl: str = "svd"

    # Global SVD defaults
    rank_k: Optional[int] = None
    rank_v: Optional[int] = None

    # Global SLERP defaults
    slerp_t: float = 0.5
    slerp_gamma: float = 1.0

    # What to merge
    merge_key: bool = True
    merge_value: bool = True

    # Layer groups
    layer_groups: List[LayerGroup] = field(default_factory=list)

    # Paged attention writeback (for Step 3)
    paged_writeback: bool = False

    # Sparse attention settings (for Step 4 - ShadowKV-style)
    enable_sparse: bool = False
    chunk_size: int = 8  # Tokens per chunk
    sparse_budget: int = 256  # Number of chunks to select
    num_outliers: int = 48  # Number of outlier chunks to keep

    # Internal: layer -> group mapping
    _layer_map: Dict[int, LayerGroup] = field(init=False, default_factory=dict)

    def __post_init__(self):
        if self.layer_merge_impl not in ("svd", "slerp"):
            raise ValueError(
                f"Invalid layer_merge_impl '{self.layer_merge_impl}'. "
                "Must be 'svd' or 'slerp'."
            )

        # Finalize each group's parameters
        if self.layer_merge_impl == "svd":
            for grp in self.layer_groups:
                grp.rank_k = grp.rank_k if grp.rank_k is not None else self.rank_k
                grp.rank_v = grp.rank_v if grp.rank_v is not None else self.rank_v
                grp.slerp_t = None
                grp.slerp_gamma = None
        else:  # "slerp"
            for grp in self.layer_groups:
                grp.slerp_t = grp.slerp_t if grp.slerp_t is not None else self.slerp_t
                grp.slerp_gamma = grp.slerp_gamma if grp.slerp_gamma is not None else self.slerp_gamma
                grp.rank_k = None
                grp.rank_v = None

        self._layer_map = self._build_layer_map()

        if self.num_layers is not None:
            for grp in self.layer_groups:
                for lyr in grp.layers:
                    if lyr >= self.num_layers:
                        raise ValueError(f"Layer index {lyr} exceeds num_layers={self.num_layers}")

    def _build_layer_map(self) -> Dict[int, LayerGroup]:
        """Build mapping from layer index to LayerGroup."""
        layer_map: Dict[int, LayerGroup] = {}
        for grp in self.layer_groups:
            for lyr in grp.layers:
                if lyr in layer_map:
                    raise ValueError(f"Layer {lyr} appears in multiple groups")
                layer_map[lyr] = grp
        return layer_map

    def get_group_for_layer(self, layer_idx: int) -> Optional[LayerGroup]:
        """Return the LayerGroup for the given layer index, or None if not found."""
        return self._layer_map.get(layer_idx, None)


def generate_consecutive_layer_groups(
    start_layer: int,
    end_layer: int,
    group_size: int,
) -> List[LayerGroup]:
    """Generate consecutive layer groups."""
    groups = []
    current = start_layer
    while current <= end_layer:
        grp_end = min(current + group_size - 1, end_layer)
        groups.append(LayerGroup(layers=list(range(current, grp_end + 1))))
        current = grp_end + 1
    return groups


def generate_consecutive_xKV_config(
    layer_merge_impl: str = "svd",
    start_layer: int = 0,
    end_layer: int = 31,
    num_layers: Optional[int] = None,
    group_size: int = 2,
    rank_k: Optional[int] = 256,
    rank_v: Optional[int] = 768,
    slerp_t: float = 0.5,
    slerp_gamma: float = 1.0,
    merge_key: bool = True,
    merge_value: bool = True,
    paged_writeback: bool = False,
    # Sparse attention parameters (Step 4)
    enable_sparse: bool = False,
    chunk_size: int = 8,
    sparse_budget: int = 256,
    num_outliers: int = 48,
) -> xKVConfig:
    """Quickly build a xKVConfig with consecutive-layer groups."""
    if end_layer == -1:
        assert num_layers is not None, "Must provide num_layers if end_layer is -1"
        end_layer = num_layers - 1

    layer_groups = generate_consecutive_layer_groups(start_layer, end_layer, group_size)

    return xKVConfig(
        num_layers=num_layers,
        layer_merge_impl=layer_merge_impl,
        rank_k=rank_k,
        rank_v=rank_v,
        slerp_t=slerp_t,
        slerp_gamma=slerp_gamma,
        merge_key=merge_key,
        merge_value=merge_value,
        layer_groups=layer_groups,
        paged_writeback=paged_writeback,
        enable_sparse=enable_sparse,
        chunk_size=chunk_size,
        sparse_budget=sparse_budget,
        num_outliers=num_outliers,
    )

