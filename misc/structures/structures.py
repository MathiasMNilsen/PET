"""
Core PET data structures.

This module defines `PETDataFrame`, a pandas `DataFrame` subclass for
ensemble-style tabular data, and `PETStateArray`, a NumPy `ndarray`
subclass for state vectors with PET-specific indexing metadata.
"""

import pandas as pd
import numpy  as np

from geostat.decomp import Cholesky
from pandas._typing import Axes, Dtype
from numpy._typing  import ArrayLike

__author__ = 'Mathias Methlie Nilsen'

__all__ = [
    'PETDataFrame',
    'PETStateArray',
]

class PETDataFrame(pd.DataFrame):
    """
    Pandas DataFrame subclass that preserves all pandas behavior
    while allowing project-specific custom methods.
    """

    # Custom attributes to preserve across pandas operations
    _metadata = ['name', 'is_ensemble'] 

    @property
    def _constructor(self):
        # Ensures pandas ops (copy, loc filtering, arithmetic, etc.)
        # return this subclass when possible.
        return PETDataFrame

    def __init__(
        self,
        data=None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
        name: str | None = None,
        is_ensemble: bool = False,  # Optional flag to indicate if this DataFrame is an ensemble
    ) -> None:
        
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.name = name
        self.is_ensemble = is_ensemble

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, name: str | None = None, is_ensemble: bool = False) -> "PETDataFrame":
        """Create a PETDataFrame from an existing pd.DataFrame."""
        out = cls(data=df, name=name, is_ensemble=is_ensemble)
        out.index.name = df.index.name
        out.attrs = df.attrs.copy()
        return out
    
    @classmethod
    def from_pickle(cls, filepath: str) -> "PETDataFrame":
        """Load a PETDataFrame from a pickle file."""
        df = pd.read_pickle(filepath)
        df.where(pd.notnull(df), None)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Pickle file {filepath} does not contain a DataFrame.")
        return cls.from_pandas(df)
    
    @classmethod
    def from_csv(cls, filepath: str, **kwargs) -> "PETDataFrame":
        """Load a PETDataFrame from a CSV file."""
        df = pd.read_csv(filepath, **kwargs)
        df.where(pd.notnull(df), None)
        return cls.from_pandas(df)

    @classmethod
    def merge_dataframes(cls, dfs: list[pd.DataFrame]) -> "PETDataFrame":
        '''
        Combine a list of DataFrames (one per ensemble member) into a single
        PETDataFrame where each cell contains an array of ensemble values.
        '''
        if len(dfs) == 0:
            raise ValueError('dfs must contain at least one DataFrame.')
        if not all(isinstance(df, pd.DataFrame) for df in dfs):
            raise ValueError('All elements in dfs must be pandas DataFrames.')

        first = dfs[0]
        for i, dfn in enumerate(dfs[1:], start=1):
            if not dfn.index.equals(first.index):
                raise ValueError(f'DataFrame at position {i} has a different index.')
            if not dfn.columns.equals(first.columns):
                raise ValueError(f'DataFrame at position {i} has different columns.')

        merged = pd.DataFrame(index=first.index, columns=first.columns, dtype=object)
        merged.index.name = first.index.name

        for idx in merged.index:
            for col in merged.columns:
                values = [dfn.at[idx, col] for dfn in dfs]
                merged.at[idx, col] = np.asarray(values).squeeze().T

        out = cls.from_pandas(merged, name=getattr(first, 'name', None), is_ensemble=True)
        out.attrs = first.attrs.copy()
        return out
    

    def to_series(self) -> pd.Series:
        mult_index = []
        for idx in self.index:
            for col in self.columns:
                mult_index.append((idx, col))
        mult_index = pd.MultiIndex.from_tuples(mult_index, names=[self.index.name, 'datatype'])
        
        values = []
        for idx in self.index:
            for col in self.columns:
                values.append(self.loc[idx, col])
        
        return pd.Series(values, index=mult_index)
    

    def to_matrix(self, filter=True, is_jacobian=False, squeeze=True) -> np.ndarray:
        
        # If multi-index columns, convert to single-level first
        if isinstance(self.columns, pd.MultiIndex):
            df = self._to_singlelevel_columns()
        else:
            df = self

        arr = []
        for val in df.to_series().values:
            if filter and (val is None or np.all(np.asarray(val) == None)):
                continue

            if (not self.is_ensemble) and isinstance(val, np.ndarray) and (not is_jacobian):
                arr.extend(val)
            else:
                arr.append(val)
        
        if is_jacobian:
            arr = np.stack(arr, axis=0)
        else:
            arr = np.row_stack(arr)

        return np.squeeze(arr) if squeeze else arr


    def _to_singlelevel_columns(self) -> "PETDataFrame":
        """
        Convert a MultiIndex-column DataFrame with structure (key, param)
        into a DataFrame with one column per key, where the value is
        the concatenation of all param-arrays for that key.
        """
        result = {}
        keys = pd.Index(self.columns.get_level_values(0)).unique()

        for key in keys:
            param_arrays = self[key]
            concatenated = [
                np.concatenate(param_arrays.iloc[i].values)
                for i in range(len(self))
            ]
            result[key] = concatenated
        
        df_new = PETDataFrame(result, index=self.index)
        df_new.index.name = self.index.name
        return df_new



class PETStateArray(np.ndarray):       

    def __new__(cls, a: ArrayLike, indices: dict[str, tuple[int, int]] | None = None) -> "PETStateArray":
        '''
        State array for Python Ensemble Toolbox.
        Works like a regular numpy array, but with extra functionality. 
        '''
        obj = np.asarray(a).view(cls)
        obj.indices = indices
        obj.state_axis = 0  # axis that holds the state variables
        return obj

    def __array_finalize__(self, obj):
        # Called on every new StateArray: construction, slicing, view, etc.
        if obj is None:
            return
        
        self.indices = getattr(obj, 'indices', None)
        self.state_axis = getattr(obj, 'state_axis', 0)

    def __repr__(self):
        return f"StateArray({np.array_repr(np.asarray(self))})"


    # --- typed operator overrides so Pylance infers PETStateArray, not ndarray ---
    def _wrap(self, result: np.ndarray) -> "PETStateArray":
        """View result as PETStateArray and carry indices and state_axis over."""
        out = result.view(PETStateArray)
        out.indices = self.indices
        out.state_axis = self.state_axis
        return out
    
    @classmethod
    def from_dict(cls, member: dict[str, np.ndarray], ne: int = None) -> "PETStateArray":
        '''
        Convert a single dictionary of state-key -> array into a PETStateArray.
        If ne is provided, only the first ne columns of each array are used.
        '''
        if len(member) == 0:
            raise ValueError('member must not be empty')

        keys = list(member.keys())

        running = 0
        indices: dict[str, tuple[int, int]] = {}
        parts: list[np.ndarray] = []

        for key in keys:
            if ne is None:
                values = np.asarray(member[key])
            else:
                values = np.asarray(member[key])[:,:ne]

            size = values.shape[0]
            indices[key] = (running, running + size)
            running += size
            parts.append(values)

        data = np.concatenate(parts)
        return cls(data, indices=indices)

    @classmethod
    def from_list_of_dicts(cls, members: list[dict[str, np.ndarray]]) -> "PETStateArray":
        '''
        Inverse of PETStateArray.to_list_of_dicts().

        Parameters
        ----------
        members:
            One dict per ensemble member. Each dict maps state-key -> 1D array.
        '''
        if len(members) == 0:
            raise ValueError('members must contain at least one dictionary')

        first = members[0]
        if len(first) == 0:
            raise ValueError('member dictionaries must not be empty')

        keys = list(first.keys())

        running = 0
        indices: dict[str, tuple[int, int]] = {}
        for key in keys:
            size = np.asarray(first[key]).shape[0]
            indices[key] = (running, running + size)
            running += size

        ne = len(members)
        nx = max(end for _, end in indices.values())
        dtype = np.asarray(first[keys[0]]).dtype
        data = np.empty((nx, ne), dtype=dtype)

        expected_keys = set(indices.keys())
        for member_index, member in enumerate(members):
            if set(member.keys()) != expected_keys:
                raise ValueError('all members must have the same keys as indices')

            for key, (start, end) in indices.items():
                values = np.asarray(member[key])
                if values.ndim != 1:
                    raise ValueError(f"member[{member_index}]['{key}'] must be 1D")
                if values.shape[0] != (end - start):
                    raise ValueError(
                        f"member[{member_index}]['{key}'] has length {values.shape[0]}, "
                        f'expected {end - start}'
                    )
                data[start:end, member_index] = values

        return cls(data.squeeze(), indices=indices)


    @classmethod
    def generate_from_prior_info(cls, prior_info: dict[str, np.ndarray], ne: int, save: bool = True) -> "PETStateArray":
        '''
        Generate a prior ensemble based on the provided prior_info dictionary.
        
        Parameters
        ----------
        prior_info : dict
            Dictionary containing prior information for each state variable.
        
        ne : int
            Number of ensemble members to generate.
        
        save : bool, optional
            Whether to save the generated ensemble to a file. Default is True.
        
        Returns
        -------
        PETStateArray
            Generated prior ensemble as a PETStateArray.
        '''
        # Initialize empty array and indices
        enX = None
        idX = {}

        # Loop over each variable in prior_info
        for name, info in prior_info.items():
            mean = info['mean']
            var  = info['variance'] 
            nx = info.get('nx', 0)
            ny = info.get('ny', 0)
            nz = info.get('nz', 0)
            
            # If no dimensions are given, nothing is generated for this variable
            if nx == ny == 0:
                break

            j = 0
            for z in range(nz):

                if isinstance(mean, (list, np.ndarray)) and len(mean) > 1:
                    # Generate covariance matrix
                    cov = Cholesky().gen_cov2d(
                        x_size = nx, 
                        y_size = ny, 
                        variance = var[z], 
                        var_range = info['corr_length'][z], 
                        aspect = info['aniso'][z], 
                        angle = info['angle'][z], 
                        var_type = info['vario'][z],
                    )
                else:
                    cov = np.array(var[z])

                i = j
                j = int((z + 1)*(len(mean)/nz))
                meanz = mean[i:j]

                # Generate ensemble members for this variable
                if info.get('limits', None) is None:
                    fieldz = Cholesky().gen_real(meanz, cov, ne)
                else:
                    fieldz = Cholesky().gen_real(meanz, cov, ne, limits=info['limits'][z])

                if z == 0:
                    field = fieldz
                else:
                    field = np.vstack((field, fieldz))
        
            # Fill in the StateArray data and indices
            if enX is None:
                enX = field
                idX[name] = (0, field.shape[0])
            else:
                enX = np.vstack((enX, field))
                idX[name] = (idX[name][0], idX[name][0] + field.shape[0])

        # Make StateArray and save
        enX = cls(enX, indices=idX)
        if save:
            np.savez('prior_ensemble.npz', **enX.to_dict())

        return enX


    # --- shape-changing ops with updated indices/state_axis ---
    @property
    def T(self) -> "PETStateArray":                          # type: ignore[override]
        out = np.asarray(self).T.view(PETStateArray)
        out.indices = self.indices
        # flip state axis: 0↔1 for 2D, generalises to ndim-1-axis
        out.state_axis = self.ndim - 1 - self.state_axis
        return out

    def reshape(self, *shape, **kwargs) -> "PETStateArray | np.ndarray":  # type: ignore[override]
        result = np.asarray(self).reshape(*shape, **kwargs)
        # Preserve PETStateArray only when the state dimension size is unchanged
        if result.shape[self.state_axis] == self.shape[self.state_axis]:
            out = result.view(PETStateArray)
            out.indices = self.indices
            out.state_axis = self.state_axis
            return out
        return result

    def ravel(self, order='C') -> np.ndarray:              # type: ignore[override]
        return np.asarray(self).ravel(order)

    def flatten(self, order='C') -> np.ndarray:            # type: ignore[override]
        return np.asarray(self).flatten(order)
    
    # -------------------------------------------------------------------------
    def __add__(self, other)       -> "PETStateArray": return self._wrap(np.add(self, other))
    def __radd__(self, other)      -> "PETStateArray": return self._wrap(np.add(other, self))
    def __sub__(self, other)       -> "PETStateArray": return self._wrap(np.subtract(self, other))
    def __rsub__(self, other)      -> "PETStateArray": return self._wrap(np.subtract(other, self))
    def __mul__(self, other)       -> "PETStateArray": return self._wrap(np.multiply(self, other))
    def __rmul__(self, other)      -> "PETStateArray": return self._wrap(np.multiply(other, self))
    def __truediv__(self, other)   -> "PETStateArray": return self._wrap(np.true_divide(self, other))
    def __rtruediv__(self, other)  -> "PETStateArray": return self._wrap(np.true_divide(other, self))
    def __floordiv__(self, other)  -> "PETStateArray": return self._wrap(np.floor_divide(self, other))
    def __pow__(self, other)       -> "PETStateArray": return self._wrap(np.power(self, other))
    def __matmul__(self, other)    -> "PETStateArray": return self._wrap(np.matmul(self, other))
    def __rmatmul__(self, other)   -> "PETStateArray": return self._wrap(np.matmul(other, self))
    def __neg__(self)              -> "PETStateArray": return self._wrap(np.negative(self))
    def __pos__(self)              -> "PETStateArray": return self._wrap(np.positive(self))
    def __abs__(self)              -> "PETStateArray": return self._wrap(np.absolute(self))
    # -------------------------------------------------------------------------


    def to_dict(self) -> dict[str, np.ndarray]:
        '''
        Convert the StateArray into a dictionary of arrays based on the provided indices.
        Slices along state_axis, so works after .T or shape-preserving .reshape.
        '''
        array = np.asarray(self)
        if self.state_axis == 0:
            return {key: array[start:end] for key, (start, end) in self.indices.items()}
        else:  # state_axis == 1, e.g. after .T on a 2D array
            return {key: array[:, start:end] for key, (start, end) in self.indices.items()}

    def to_list_of_dicts(self) -> list[dict[str, np.ndarray]]:
        '''
        Convert the StateArray into a list of dictionaries, one per ensemble member.
        Works regardless of state_axis (e.g. after .T).
        '''
        array = np.asarray(self)
        if self.state_axis == 0:
            # state on axis 0, ensemble on axis 1
            if array.ndim == 1:
                array = array[:, np.newaxis]
            ne = array.shape[1]
            slices = {key: array[start:end] for key, (start, end) in self.indices.items()}
            return [{key: slices[key][:, n] for key in slices} for n in range(ne)]
        else:
            # state on axis 1 (e.g. after .T), ensemble on axis 0
            if array.ndim == 1:
                array = array[np.newaxis, :]
            ne = array.shape[0]
            slices = {key: array[:, start:end] for key, (start, end) in self.indices.items()}
            return [{key: slices[key][n] for key in slices} for n in range(ne)]


    def clip_matrix(self, limits) -> None:
        '''
        Clip the values in the StateArray in place using the provided limits.
        
        Parameters
        ----------
        limits : dict, tuple, or list
            If tuple, it should be (lower_bound, upper_bound) applied to all variables.
            If dict, it should have variable names as keys and (lower_bound, upper_bound) as values.
            If list, it should contain (lower_bound, upper_bound) tuples for each variable in the order of indices.

        '''
        array = np.asarray(self)

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
