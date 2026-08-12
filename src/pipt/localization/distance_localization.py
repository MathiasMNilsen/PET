"""Distance-based localization implementation."""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

from pipt.localization.common import LocalizationBase
from pipt.misc_tools.extract_tools import list_to_dict

__all__ = [
    "DistanceLocalization",
    "GaspariCohnKernel",
    "FurrerBengtssonKernel",
    "RegionKernel",
]


# ===========================================================
# Localization entry container
# ===========================================================

@dataclass(slots=True)
class LocalizationEntry:
    """Configuration for a single (data_type, time, parameter) localization entry."""

    taper:            str
    positions:        List[List[int]]
    radius:           int
    z_range:          object
    anisotropy_ratio: float = 1.0
    rotation_deg:     float = 0.0
    filepath:         Optional[str] = None   # used when taper == 'import'


# ===========================================================
# Geometry helpers
# ===========================================================

def _build_transform(anisotropy_ratio: float, rotation_deg: float) -> np.ndarray:
    """Return the 2x2 anisotropy + rotation transform matrix."""
    angle    = np.deg2rad(rotation_deg)
    rotation = np.array([[ np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]])
    scaling  = np.array([[1.0 / anisotropy_ratio, 0.0],
                         [0.0,                    1.0]])
    return scaling @ rotation


def _kernel_coordinates(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) coordinate grids centered at the origin."""
    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    return np.meshgrid(x, y, indexing="ij")


def _crop_kernel(kernel: np.ndarray) -> np.ndarray:
    """Trim zero-only border rows and columns from a kernel array."""
    rows = np.any(kernel > 0, axis=1)
    cols = np.any(kernel > 0, axis=0)
    r0 = rows.argmax();  r1 = len(rows) - rows[::-1].argmax()
    c0 = cols.argmax();  c1 = len(cols) - cols[::-1].argmax()
    return kernel[r0:r1, c0:c1]


# ===========================================================
# Kernel classes
# ===========================================================

class GaspariCohnKernel:
    """Gaspari-Cohn compactly supported smooth taper kernel."""

    def build(
        self,
        radius:           int,
        anisotropy_ratio: float,
        rotation_deg:     float,
        field_shape:      tuple,
        ensemble_size:    Optional[int] = None,
    ) -> np.ndarray:
        nx, ny = 2 * field_shape[1], 2 * field_shape[2]
        X, Y   = _kernel_coordinates(nx, ny)
        coords = np.vstack((X.ravel(), Y.ravel()))

        T           = _build_transform(anisotropy_ratio, rotation_deg)
        transformed = T @ coords
        ratio       = np.sqrt((transformed[0] / radius) ** 2 +
                              (transformed[1] / radius) ** 2)

        values = np.zeros_like(ratio)
        inner  = ratio <= 1
        outer  = (ratio > 1) & (ratio <= 2)

        values[inner] = (
            -0.25 * ratio[inner] ** 5
            + 0.5  * ratio[inner] ** 4
            + 0.625 * ratio[inner] ** 3
            - (5.0 / 3.0) * ratio[inner] ** 2
            + 1.0
        )
        values[outer] = (
            (1.0 / 12.0) * ratio[outer] ** 5
            - 0.5  * ratio[outer] ** 4
            + 0.625 * ratio[outer] ** 3
            + (5.0 / 3.0) * ratio[outer] ** 2
            - 5.0  * ratio[outer]
            + 4.0
            - (2.0 / 3.0) / ratio[outer]
        )

        return _crop_kernel(values.reshape(nx, ny))


class FurrerBengtssonKernel:
    """Furrer-Bengtsson ensemble-size-aware taper kernel."""

    def build(
        self,
        radius:           int,
        anisotropy_ratio: float,
        rotation_deg:     float,
        field_shape:      tuple,
        ensemble_size:    Optional[int] = None,
    ) -> np.ndarray:
        nx, ny = 2 * field_shape[1], 2 * field_shape[2]
        X, Y   = _kernel_coordinates(nx, ny)
        coords = np.vstack((X.ravel(), Y.ravel()))

        T           = _build_transform(anisotropy_ratio, rotation_deg)
        transformed = T @ coords
        distance    = np.sqrt(transformed[0] ** 2 + transformed[1] ** 2)

        weight         = np.zeros_like(distance)
        inside         = distance < radius
        d              = distance[inside] / radius
        weight[inside] = 1.0 - (1.5 * d - 0.5 * d ** 3)

        ne = ensemble_size if ensemble_size is not None else 50
        fb = (ne * weight ** 2) / (weight ** 2 * (ne + 1) + 1)

        return _crop_kernel(fb.reshape(nx, ny))


class RegionKernel:
    """Binary region kernel - full weight (1) everywhere within range."""

    def build(
        self,
        radius:           int           = None,
        anisotropy_ratio: float         = 1.0,
        rotation_deg:     float         = 0.0,
        field_shape:      tuple         = None,
        ensemble_size:    Optional[int] = None,
    ) -> np.ndarray:
        return np.ones((1, 1))


# ===========================================================
# DistanceLocalization - mirrors AutoAdaptiveLocalization API
# ===========================================================

class DistanceLocalization(LocalizationBase):
    """
    Distance-based localization strategy for sparse mask projection.

    Follows the same init/call pattern as AutoAdaptiveLocalization:
    - All configuration is parsed and stored at ``__init__`` time.
    - ``__call__`` assembles and returns the sparse localization operator.

    Parameters
    ----------
    info : dict or list
        Localization configuration. Must contain:

        - ``field``: ``[nz, nx, ny]`` grid dimensions.
        - ``actnum``: path to ``.npz`` file with active-cell mask (optional).
        - ``taper_func``: kernel type -- ``"gc"``, ``"fb"``, or ``"region"``
          (default: ``"region"``).

        Plus one of:
        - a ``.csv`` key or comma-separated inline rows specifying entries, or
        - a ``.pkl`` / ``.p`` key pointing to a pre-built entries dict.

    data : pd.DataFrame, optional
        Observed data with time indices as rows and data types as columns.

    parameters : list of str, optional
        State parameter names used as defaults in ``__call__``.

    ensemble_size : int, optional
        Ensemble size; used by the Furrer-Bengtsson kernel.

    prior_info : dict, optional
        Per-parameter prior information (``nx``, ``ny``, ``nz``).
        Used to build zero masks for unconfigured parameters.
    """

    name = "distance_loc"

    _kernel_map = {
        "gc":     GaspariCohnKernel,
        "fb":     FurrerBengtssonKernel,
        "region": RegionKernel,
    }

    def __init__(
        self,
        info:          Union[dict, list],
        data:          Union[pd.DataFrame, None] = None,
        parameters:    Union[list, None]         = None,
        ensemble_size: Union[int, None]          = None,
        prior_info:    Union[dict, None]         = None,
    ):
        """
        Initialize the DistanceLocalization instance.

        Spatial localization entries — one per (data_type, time, parameter)
        combination — are supplied either via an external CSV file or as
        comma-separated inline rows embedded in the ``info`` dict key.
        All ``info`` keys map directly to the ``[dataassim.localization]``
        table in a TOML config file.

        Parameters
        ----------
        info : dict or list
            Localization configuration. Recognised keys:

            **field** : list of int, *required*
                Grid dimensions ``[nz, nx, ny]``. Used to size the spatial
                kernel arrays and to lay out the flattened cell vectors.

            **actnum** : str, *optional*
                Path to a ``.npz`` file whose first array is a boolean mask
                of active cells. When supplied, only active cells appear in
                the output localization operator. Default: ``None``.

            **taper_func** : {``"gc"``, ``"fb"``, ``"region"``}, *optional*
                Spatial kernel applied at each observation location:

                - ``"gc"`` — **Gaspari-Cohn** fifth-order piecewise
                  polynomial. Compact support extends to ``2 × radius``
                  grid cells. Values lie in [0, 1] with a smooth,
                  differentiable profile. The standard choice for
                  distance-based localization in geoscience DA.
                - ``"fb"`` — **Furrer-Bengtsson** ensemble-size-aware
                  taper. Weights are scaled by ensemble size *Ne* so
                  that larger ensembles produce sharper localization.
                  Values lie in [0, Ne/(Ne+2)]. Pass ``ensemble_size``
                  to control *Ne* (default 50).
                - ``"region"`` — Binary point kernel: weight 1 at the
                  single nearest cell, 0 everywhere else. Equivalent to
                  assigning one observation to exactly one grid cell.

                Default: ``"region"``.

            **<filename>.csv** : any, *optional*
                A key whose name ends in ``.csv`` is opened as a path to
                a CSV file containing one localization entry per line
                (see *CSV row format* in Notes). The associated value is
                ignored. Recommended for configurations with many entries.

            **"<row1>,<row2>,..."** : any, *optional*
                Any key containing a comma is split on ``,`` and each
                segment is parsed as a localization entry row. Useful for
                small configurations that do not need an external file.

            **<filename>.pkl / .p** : any, *optional*
                A key ending in ``.pkl`` or ``.p`` is loaded with
                :func:`pickle.load` and must contain a pre-built
                ``{(data_type, time, param): LocalizationEntry}`` dict.
                Intended for offline pre-computation of expensive masks.

        data : pd.DataFrame, optional
            Observed data whose **index** contains the assimilation time
            steps (must match the ``time`` field in each CSV row) and
            whose **columns** are the data-type names (e.g.
            ``"WOPR PRO1"``). Required for ``__call__`` to produce output.

        parameters : list of str, optional
            Ordered list of state parameter names (e.g.
            ``["permx", "poro"]``). Determines which parameters receive
            a localization mask and the stacking order in the output.

        ensemble_size : int, optional
            Ensemble size *Ne*. Only affects the Furrer-Bengtsson kernel
            (``taper_func = "fb"``). Default: ``None`` (``"fb"`` falls
            back to *Ne* = 50).

        prior_info : dict, optional
            Per-parameter grid sizes. Required only when a parameter
            appears in ``parameters`` but has **no** localization entry
            in the CSV; such parameters receive an all-zero weight column
            whose length is taken from this dict::

                {"poro": {"nx": 20, "ny": 20, "nz": 1}}

        Notes
        -----
        **CSV row format**

        Each entry is a single space-separated line with 11 fields
        (or 12 if the data-type name contains a space)::

            taper  x_pos  y_pos  z_pos  radius  z_range  aniso  rotation  data_type  time  param

        For two-word data types (e.g. ``WOPR PRO1``) use 12 fields::

            taper  x_pos  y_pos  z_pos  radius  z_range  aniso  rotation  word1  word2  time  param

        Field descriptions:

        - **taper** — kernel tag: ``gc``, ``fb``, or ``region``.
        - **x_pos** — observation x-cell index on the grid (0-based), along the ``nx`` axis.
        - **y_pos** — observation y-cell index on the grid (0-based), along the ``ny`` axis.
        - **z_pos** — observation layer index on the grid (0-based).
        - **radius** — kernel half-radius in grid cells. For ``gc`` the
          full support spans ``2 × radius`` cells from the center.
        - **z_range** — ``":"`` to spread the kernel across all *nz*
          layers, or an integer to restrict it to that single layer.
        - **aniso** — anisotropy ratio (x-axis scaling factor). Use
          ``1.0`` for isotropic kernels; ``2.0`` compresses the kernel
          to half-width in the x-direction.
        - **rotation** — clockwise rotation of the kernel in degrees.
          Use ``0.0`` for axis-aligned kernels.
        - **data_type** — observation type name, case-insensitive. Must
          match a column in the ``data`` DataFrame.
        - **time** — assimilation time step; must match an index value
          of the ``data`` DataFrame.
        - **param** — state parameter name, case-insensitive. Must appear
          in the ``parameters`` list.

        Examples
        --------
        TOML config using an external CSV file (recommended for many
        observation types or time steps):

        ```toml
        [dataassim.localization]
        name       = "distance_loc"
        field      = [1, 20, 20]    # [nz, nx, ny]
        taper_func = "gc"
        "loc_entries.csv" = true    # key = filename; value is ignored
        ```

        Example ``loc_entries.csv`` (Gaspari-Cohn, isotropic, all layers):

        ```
        gc 10 10 0  6 : 1.0  0.0  pressure    400.0 permx
        gc 10 10 0  6 : 1.0  0.0  pressure    800.0 permx
        gc  5 15 0  4 : 1.0  0.0  wopr pro1   400.0 permx
        gc  5 15 0  4 : 2.0 30.0  wopr pro1   800.0 permx
        ```

        TOML config using the Furrer-Bengtsson kernel with active-cell
        mask and anisotropic entries in the CSV:

        ```toml
        [dataassim.localization]
        name       = "distance_loc"
        field      = [2, 30, 40]    # two-layer, 30×40 lateral grid
        taper_func = "fb"
        actnum     = "active.npz"
        "loc_entries.csv" = true
        ```

        ``loc_entries.csv`` restricting each observation to layer 0 only
        (``z_range = 0``) with anisotropic, rotated kernel:

        ```
        fb  8 12 0 6 0 2.0 45.0 wopr pro1 400.0 permx
        fb 15  5 0 8 0 1.0  0.0 wwct pro2 400.0 permx
        ```
        """
        if isinstance(info, list):
            info = list_to_dict(info)

        # -- shared config (field shape + actnum) from base class
        self.field, self.actnum = self.config_common(info)

        # -- store all call-time defaults as instance attributes
        self.parameters    = parameters
        self.prior_info    = prior_info if prior_info is not None else {}
        self.ensemble_size = ensemble_size

        # -- select and instantiate the kernel
        taperfunc = info.get("taper_func", "region")
        if taperfunc not in self._kernel_map:
            raise ValueError(
                f"Unknown taper_func '{taperfunc}'. "
                f"Supported: {list(self._kernel_map)}"
            )
        self._kernel = self._kernel_map[taperfunc]()

        # -- data and derived index/type lists
        self.data = data
        if self.data is not None:
            self.data_indices = list(self.data.index)
            self.data_types   = list(self.data.columns)
        else:
            self.data_indices = None
            self.data_types   = None

        # -- parse config entries and precompute kernel masks
        self._entries: Dict[Tuple, LocalizationEntry] = (
            self._parse_config(info) if self.data_indices is not None else {}
        )
        self._mask_cache: Dict[tuple, np.ndarray] = self._build_mask_cache()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        curr_data:  Union[list, None] = None,
        curr_time:  Union[list, None] = None,
        curr_param: Union[list, None] = None,
    ) -> sparse.spmatrix:
        """
        Build the sparse localization operator for the current assimilation step.

        Parameters
        ----------
        curr_data : list of str, optional
            Data types to include. Defaults to ``self.data_types``.
        curr_time : list, optional
            Time indices to include. Defaults to ``self.data_indices``.
        curr_param : list of str, optional
            State parameters to update. Defaults to ``self.parameters``.

        Returns
        -------
        scipy.sparse matrix, shape (n_active_cells, n_obs)
            Sparse localization operator.
        """
        curr_data  = self.data_types   if curr_data  is None else curr_data
        curr_time  = self.data_indices if curr_time  is None else curr_time
        curr_param = self.parameters   if curr_param is None else curr_param

        loc_blocks = []

        for time in curr_time:
            for data_name in curr_data:

                cell  = self.data.loc[time, data_name]
                n_obs = len(cell) if hasattr(cell, "__len__") else 1
                if n_obs <= 0:
                    continue

                obs_blocks = [[] for _ in range(n_obs)]

                for param in curr_param:
                    key = (data_name, time, param)
                    if key in self._entries and self._entries[key].taper is not None:
                        mask = self._resolve_mask(key)
                        for i in range(n_obs):
                            obs_blocks[i].append(mask)
                    else:
                        zero_masks = self._zero_mask(param, n_obs)
                        for i in range(n_obs):
                            obs_blocks[i].append(zero_masks[i])

                for blocks in obs_blocks:
                    sparse_blocks = [sparse.csc_matrix(b.reshape(1, -1)) for b in blocks]
                    loc_blocks.append(
                        sparse.hstack(sparse_blocks) if len(sparse_blocks) > 1
                        else sparse_blocks[0]
                    )

        return sparse.vstack(loc_blocks).transpose()

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    def _parse_config(self, info: dict) -> Dict[Tuple, LocalizationEntry]:
        """Parse localization config into a ``(data_type, time, param)`` entry dict."""

        # -- pickle shortcut
        for v in info.values():
            if str(v).endswith((".p", ".pkl")):
                with open(v, "rb") as f:
                    raw = pickle.load(f)
                return {k: v for k, v in raw.items()
                        if isinstance(k, tuple) and len(k) == 3}

        # -- skeleton: one empty entry per (data_type, time, param) combo
        entries: Dict[Tuple, LocalizationEntry] = {
            (datum, time, param): LocalizationEntry(
                taper=None, positions=None, radius=None, z_range=None
            )
            for time  in self.data_indices
            for datum in self.data_types
            for param in self.parameters
        }

        # -- read rows from CSV file or inline comma-separated string
        csv_key = next((k for k in info if str(k).endswith(".csv")), None)
        if csv_key:
            with open(csv_key) as f:
                rows = [item for sublist in csv.reader(f) for item in sublist]
        else:
            rows = next(
                (str(k).split(",") for k in info if len(str(k).split(",")) > 1),
                [],
            )

        for row in rows:
            parts = row.split()
            if not parts:
                continue

            # ── import rows: "import filename z_range data_type time param"
            # (6 fields for single-word data type, 7 for two-word)
            if parts[0] == 'import':
                if len(parts) == 6:
                    key = (parts[3].lower(), float(parts[4]), parts[5].lower())
                else:
                    key = (f"{parts[3].lower()} {parts[4].lower()}",
                           float(parts[5]), parts[6].lower())
                if key not in entries:
                    continue
                entries[key] = LocalizationEntry(
                    taper    = 'import',
                    positions = None,
                    radius   = None,
                    z_range  = parts[2],
                    filepath = parts[1],
                )
                continue

            # ── standard rows: 11 fields (single-word) or 12 (two-word data type)
            if len(parts) == 11:
                key = (parts[8].lower(), float(parts[9]), parts[10].lower())
            else:
                key = (
                    f"{parts[8].lower()} {parts[9].lower()}",
                    float(parts[10]),
                    parts[11].lower(),
                )

            if key not in entries:
                continue

            entries[key] = LocalizationEntry(
                taper            = parts[0],
                positions        = [[int(float(parts[1])),
                                     int(float(parts[2])),
                                     int(float(parts[3]))]],
                radius           = int(parts[4]),
                z_range          = parts[5],
                anisotropy_ratio = float(parts[6]),
                rotation_deg     = float(parts[7]),
            )

        return entries

    # ------------------------------------------------------------------
    # Mask caching
    # ------------------------------------------------------------------

    def _build_mask_cache(self) -> Dict[tuple, np.ndarray]:
        """Precompute unique spatial kernel arrays for all active entries."""
        cache: Dict[tuple, np.ndarray] = {}
        for entry in self._entries.values():
            if entry.taper is None:
                continue
            key = self._cache_key(entry)
            if key not in cache:
                if entry.taper == 'import':
                    data = np.load(entry.filepath)
                    arr  = data[data.files[0]] if hasattr(data, 'files') and data.files else data
                    cache[key] = arr.reshape(self.field)   # ensure (nz, nx, ny)
                else:
                    cache[key] = self._kernel.build(
                        radius           = entry.radius,
                        anisotropy_ratio = entry.anisotropy_ratio,
                        rotation_deg     = entry.rotation_deg,
                        field_shape      = self.field,
                        ensemble_size    = self.ensemble_size,
                    )
        return cache

    @staticmethod
    def _cache_key(entry: LocalizationEntry) -> tuple:
        if entry.taper == 'import':
            return ('import', entry.filepath)
        return (entry.taper, entry.radius, entry.anisotropy_ratio, entry.rotation_deg)

    # ------------------------------------------------------------------
    # Call-time helpers
    # ------------------------------------------------------------------

    def _resolve_mask(self, key: Tuple[str, float, str]) -> np.ndarray:
        """Return the repositioned spatial mask for an entry key."""
        entry = self._entries[key]

        if entry.taper == 'import':
            # pre-computed full 3-D mask loaded from .npz
            mask = self._mask_cache[self._cache_key(entry)]  # shape: (nz, nx, ny)
        else:
            kernel = self._mask_cache[self._cache_key(entry)]
            masks  = [self._place_kernel(kernel, pos) for pos in entry.positions]
            mask   = np.maximum.reduce(masks)                 # shape: (nz, nx, ny)

        if entry.z_range != ":":
            z    = int(entry.z_range)
            flat = mask[z].flatten()                          # shape: (nx*ny,)
            if self.actnum is not None:
                nz, nx, ny    = self.field
                layer_actnum  = self.actnum.reshape(nz, nx, ny)[z].flatten()
                return flat[layer_actnum]
            return flat

        flat = mask.flatten()                                  # shape: (nz*nx*ny,)
        return flat[self.actnum] if self.actnum is not None else flat

    def _place_kernel(self, kernel: np.ndarray, position: List[int]) -> np.ndarray:
        """
        Place a compact 2-D kernel patch at ``position`` on the 3-D grid.

        Uses clip arithmetic to handle all grid edges uniformly.

        Parameters
        ----------
        kernel : np.ndarray, shape (ky, kx)
        position : [x_pos, y_pos, z_pos]
        """
        result             = np.zeros(self.field)
        nz, nx, ny         = self.field
        ky, kx             = kernel.shape
        x_pos, y_pos, z_pos = position

        x_min = x_pos - kx // 2;  x_max = x_min + kx
        y_min = y_pos - ky // 2;  y_max = y_min + ky

        gx0 = max(0, x_min);  gx1 = min(nx, x_max)
        gy0 = max(0, y_min);  gy1 = min(ny, y_max)

        kx0 = gx0 - x_min;  kx1 = kx0 + (gx1 - gx0)
        ky0 = gy0 - y_min;  ky1 = ky0 + (gy1 - gy0)

        result[z_pos, gx0:gx1, gy0:gy1] = kernel[ky0:ky1, kx0:kx1]
        return result

    def _zero_mask(self, param: str, n_obs: int) -> List[np.ndarray]:
        """Return zero-valued masks for a parameter with no localization entry."""
        p       = self.prior_info[param]
        n_cells = p["nx"] * p["ny"] * p["nz"]

        if n_obs > 1:
            mat = np.zeros((n_obs, n_cells))
            return [mat[i, self.actnum] if self.actnum is not None else mat[i]
                    for i in range(n_obs)]

        vec = np.zeros(n_cells)
        return [vec[self.actnum] if self.actnum is not None else vec]
