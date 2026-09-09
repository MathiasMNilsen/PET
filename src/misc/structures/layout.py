"""Row layout of the data vector: which observed cell owns which rows of an ``(nd, ...)`` array.

Observed data arrive as a frame with one row per report label (a time, a
date, an index) and one column per data type; a cell holds a scalar or a
vector, or nothing when that type was not observed at that label. Every
matrix the analyses work on -- the observation vector, its variance, the
predicted-data ensemble, the adjoints -- lists those cells in one fixed
order, label-major then type, skipping the empty ones. :class:`DataLayout`
is that order, computed once from the observed frame. Anything built from it
is aligned with anything else built from it by construction, which is what
the frame filters used to promise and could not keep once a cell was empty.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from misc.structures.structures import PETDataFrame

__all__ = ["DataLayout", "LayoutRow"]


def is_missing(cell) -> bool:
    """Whether a frame cell holds no observation: ``None`` or nothing but NaN."""
    if cell is None:
        return True
    return not np.any(pd.notna(np.atleast_1d(cell)))


@dataclass(frozen=True)
class LayoutRow:
    """One observed cell and the rows it owns: ``[start, stop)``."""

    label: object
    datatype: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def rows(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class DataLayout:
    """The order of the data vector, derived once from the observed frame."""

    rows: tuple
    labels: tuple
    datatypes: tuple
    label_name: object = None

    @classmethod
    def from_frame(cls, frame) -> "DataLayout":
        """Walk ``frame`` label-major then type, as the frame flatten did, skipping empty cells."""
        rows, start = [], 0
        for label in frame.index:
            for datatype in frame.columns:
                cell = frame.loc[label, datatype]
                if is_missing(cell):
                    continue
                size = int(np.size(cell))
                rows.append(LayoutRow(label, datatype, start, start + size))
                start += size
        return cls(tuple(rows), tuple(frame.index), tuple(frame.columns), frame.index.name)

    @property
    def nd(self) -> int:
        return self.rows[-1].stop if self.rows else 0

    def row(self, label, datatype) -> LayoutRow:
        for row in self.rows:
            if row.label == label and row.datatype == datatype:
                return row
        raise KeyError(f"no observed cell at ({label!r}, {datatype!r})")

    def row_datatypes(self) -> np.ndarray:
        """The data type of every row of the vector, ``(nd,)``."""
        return np.array([row.datatype for row in self.rows for _ in range(row.size)], dtype=object)

    # ------------------------------------------------------------------
    # Frame -> array
    # ------------------------------------------------------------------
    def vector(self, frame) -> np.ndarray:
        """The observed cells of ``frame`` as an ``(nd,)`` vector, in layout order."""
        out = np.empty(self.nd)
        for row in self.rows:
            out[row.rows] = np.ravel(np.asarray(frame.loc[row.label, row.datatype], dtype=float))
        return out

    def matrix(self, frame, ne) -> np.ndarray:
        """The cells of an ensemble ``frame`` -- ``(ne,)`` or ``(size, ne)`` each -- as ``(nd, ne)``."""
        out = np.empty((self.nd, ne))
        for row in self.rows:
            out[row.rows, :] = np.asarray(frame.loc[row.label, row.datatype], dtype=float).reshape(row.size, ne)
        return out

    # ------------------------------------------------------------------
    # Array -> frame (the view)
    # ------------------------------------------------------------------
    def to_frame(self, values, name=None) -> PETDataFrame:
        """A frame view of ``values`` -- ``(nd,)`` or ``(nd, ne)`` -- with empty cells ``None``.

        Cells come out as the flatten expects them back: a scalar for a
        one-row observation, a vector or an ``(size, ne)`` block otherwise.
        """
        values = np.asarray(values)
        frame = pd.DataFrame({datatype: [None] * len(self.labels) for datatype in self.datatypes},
                             index=pd.Index(self.labels, name=self.label_name), dtype=object)
        for row in self.rows:
            block = values[row.rows]
            if row.size == 1:
                block = float(block[0]) if values.ndim == 1 else block[0]   # a scalar, or its (ne,) ensemble
            frame.at[row.label, row.datatype] = block
        return PETDataFrame.from_pandas(frame, name=name, is_ensemble=values.ndim == 2)
