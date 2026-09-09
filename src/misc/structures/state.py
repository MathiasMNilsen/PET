"""The state's variable layout: which rows of an ``(nx, ne)`` state matrix belong to which variable.

The state itself is a plain array. The ``{name: (start, stop)}`` map used to
ride on an ``ndarray`` subclass, copied onto every slice and view (wrongly:
a slice of five rows still claimed the full layout) and lost on unpickling.
It now lives once, as the ensemble's ``idX`` dictionary, and this class gives
it the conversions the boundary needs: one dictionary per variable for
saving and QA/QC, one dictionary per member for the simulator, clipping to
the prior's limits, and the two constructors that build a state, from a
dictionary of arrays or from the prior description.
"""

from dataclasses import dataclass

import numpy as np

from misc.sampling import gen_real

__all__ = ["StateLayout"]


def _gen_real_limits(limits, layer):
    """Translate a prior's ``limits`` entry into what ``gen_real`` expects.

    Configs give ``limits`` as a single ``[lower, upper]`` pair -- the form
    the update-step clipping and :func:`limit_state` also read -- while
    ``gen_real`` wants a ``{'lower': ..., 'upper': ...}`` mapping. A per-layer
    list of either form is accepted too, for a prior that bounds its layers
    differently.
    """
    if isinstance(limits, dict):
        entry = limits
    elif isinstance(limits[0], (list, tuple, dict)):
        entry = limits[layer]
    else:
        entry = limits
    if isinstance(entry, dict):
        return entry
    lower, upper = entry
    return {'lower': lower, 'upper': upper}


@dataclass(frozen=True)
class StateLayout:
    """Row ranges of the state variables in an ``(nx, ne)`` state matrix, in stacking order."""

    indices: dict

    @property
    def nx(self) -> int:
        return max((stop for _, stop in self.indices.values()), default=0)

    @property
    def variables(self) -> tuple:
        return tuple(self.indices)

    def rows(self, name) -> slice:
        start, stop = self.indices[name]
        return slice(start, stop)

    # ------------------------------------------------------------------
    # Constructors: a state matrix and its layout
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, member, ne=None):
        """Stack ``{variable: (n, ne) array}`` into a state matrix; returns ``(matrix, layout)``.

        With ``ne`` given, only the first ``ne`` columns of each array are used.
        """
        if len(member) == 0:
            raise ValueError('member must not be empty')
        running, indices, parts = 0, {}, []
        for key, values in member.items():
            values = np.asarray(values) if ne is None else np.asarray(values)[:, :int(ne)]
            indices[key] = (running, running + values.shape[0])
            running += values.shape[0]
            parts.append(values)
        return np.concatenate(parts), cls(indices)

    @classmethod
    def from_prior_info(cls, prior_info, ne, rng=None, save=True):
        """Draw a prior ensemble from the prior description; returns ``(matrix, layout)``.

        Parameters
        ----------
        prior_info : dict
            Per variable: ``mean``, ``variance`` (per layer), the grid size
            ``nx``/``ny``/``nz`` and, for fields, the covariance description.
        ne : int
            Number of members.
        rng : RandomState-like, optional
            The stream to draw from; the global one by default.
        save : bool, optional
            Write the prior to ``prior_ensemble.npz`` (default True).
        """
        from geostat.decomp import Cholesky

        enX, idX = None, {}
        for name, info in prior_info.items():
            mean = info['mean']
            var = info['variance']
            nx, ny, nz = info.get('nx', 0), info.get('ny', 0), info.get('nz', 0)
            if nx == ny == 0:
                break

            j = 0
            field = None
            for z in range(nz):
                if isinstance(mean, (list, np.ndarray)) and len(mean) > 1:
                    cov = Cholesky().gen_cov2d(
                        x_size=nx, y_size=ny, variance=var[z], var_range=info['corr_length'][z],
                        aspect=info['aniso'][z], angle=info['angle'][z], var_type=info['vario'][z],
                    )
                else:
                    cov = np.array(var[z])

                i = j
                j = int((z + 1) * (len(mean) / nz))
                meanz = mean[i:j]

                if info.get('limits', None) is None:
                    fieldz = gen_real(meanz, cov, ne, rng=rng)
                else:
                    fieldz = gen_real(meanz, cov, ne, rng=rng, limits=_gen_real_limits(info['limits'], z))
                field = fieldz if field is None else np.vstack((field, fieldz))

            if enX is None:
                enX = field
                idX[name] = (0, field.shape[0])
            else:
                start = enX.shape[0]
                enX = np.vstack((enX, field))
                idX[name] = (start, start + field.shape[0])

        layout = cls(idX)
        if save:
            np.savez('prior_ensemble.npz', **layout.to_dict(enX))
        return enX, layout

    # ------------------------------------------------------------------
    # Conversions at the boundary
    # ------------------------------------------------------------------
    def to_dict(self, matrix) -> dict:
        """``{variable: rows}`` views of ``matrix``."""
        array = np.asarray(matrix)
        return {key: array[start:stop] for key, (start, stop) in self.indices.items()}

    def member_dicts(self, matrix) -> list:
        """One ``{variable: values}`` per member -- what a simulator takes."""
        array = np.asarray(matrix)
        if array.ndim == 1:
            array = array[:, np.newaxis]
        slices = {key: array[start:stop] for key, (start, stop) in self.indices.items()}
        return [{key: slices[key][:, n] for key in slices} for n in range(array.shape[1])]

    def clip(self, matrix, limits) -> None:
        """Clip ``matrix`` in place to ``limits``.

        ``limits`` is a ``(lower, upper)`` pair for every variable, a
        ``{variable: (lower, upper)}`` dict, or a list of pairs in stacking
        order; ``None`` bounds are left open.
        """
        array = np.asarray(matrix)
        if isinstance(limits, tuple):
            lb, ub = limits
            if not (lb is None and ub is None):
                np.clip(array, lb, ub, out=array)
        elif isinstance(limits, dict):
            for key, (i, j) in self.indices.items():
                if key in limits:
                    lb, ub = limits[key]
                    if not (lb is None and ub is None):
                        np.clip(array[i:j], lb, ub, out=array[i:j])
        elif isinstance(limits, list):
            for (key, (i, j)), (lb, ub) in zip(self.indices.items(), limits):
                if not (lb is None and ub is None):
                    np.clip(array[i:j], lb, ub, out=array[i:j])
        else:
            raise ValueError("limits must be a tuple, dict, or list")
