'''
Tests for PET structures (PETDataFrame, PETStateArray) and their methods.
'''
import pytest
import numpy as np
import pandas as pd

from misc.structures.structures import PETDataFrame, PETStateArray


# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

nparams = 3 # number of parameters
nx = 8 # state dimension
nr = 2 # nrows
nc = 3 # ncols
ny = nr*nc
ne = 10 # ensemble members

# Generate ne mulit-column dataframes with random data for testing
np.random.seed(404)  # For reproducibility
mi_dfs = []
for n in range(ne):
    data_dict = {
        ("keyA", "param1"): [np.random.rand(nx) for _ in range(nr)],
        ("keyA", "param2"): [np.random.rand(nx) for _ in range(nr)],
        ("keyA", "param3"): [np.random.rand(nx) for _ in range(nr)],
        ("keyB", "param1"): [np.random.rand(nx) for _ in range(nr)],
        ("keyB", "param2"): [np.random.rand(nx) for _ in range(nr)],
        ("keyB", "param3"): [np.random.rand(nx) for _ in range(nr)],
        ("keyC", "param1"): [np.random.rand(nx) for _ in range(nr)],
        ("keyC", "param2"): [np.random.rand(nx) for _ in range(nr)],
        ("keyC", "param3"): [np.random.rand(nx) for _ in range(nr)],
    }

    cols = pd.MultiIndex.from_tuples(data_dict.keys())
    df = pd.DataFrame(data_dict, columns=cols, index=['row1', 'row2'])
    df.index.name = "index"
    mi_dfs.append(df)


# ==============================================================================
# PETDataFrame TESTS
# ==============================================================================

@pytest.fixture
def multicolumn_dataframe():
    return mi_dfs[0]

@pytest.fixture
def ensemble_dataframe():
    pdfs = [PETDataFrame._to_singlelevel_columns(df) for df in mi_dfs]
    return pdfs

@pytest.fixture
def multicolumn_ensemble_dataframe():
    return mi_dfs


class TestSimplePETDataFrame:

    # Create a simple DataFrame for testing
    data_dict = {
        'keyA': [1.0, 2.0],
        'keyB': [3.0, 4.0],
        'keyC': [5.0, 6.0]
    }
    units = {key: f'unit:{key}' for key in data_dict.keys()}
    index = ['idx1', 'idx2']
    index_name = 'index'
    simple_df = pd.DataFrame(data_dict, index=index)
    simple_df.index.name = index_name
    simple_df.attrs['units'] = units

    def test_from_pandas(self):
        '''Test that PETDataFrame can be created from a pandas DataFrame and that the data is preserved.'''
        pdf_from_pandas = PETDataFrame.from_pandas(self.simple_df)
        pdf = PETDataFrame(data=self.data_dict, index=self.index)
        pdf.index.name = self.index_name
        assert isinstance(pdf_from_pandas, PETDataFrame)
        assert pdf_from_pandas.equals(pdf)

    def test_attrs_preserved(self):
        '''Test that attributes from the original pandas DataFrame are preserved in the PETDataFrame.'''
        pdf = PETDataFrame.from_pandas(self.simple_df)
        assert pdf.attrs['units'] == self.units

    def test_to_matrix(self):
        '''Test that the to_matrix method correctly converts the PETDataFrame to a numpy array.'''
        pdf = PETDataFrame.from_pandas(self.simple_df)
        vec = pdf.to_matrix(squeeze=False)
        vec_squeezed = pdf.to_matrix(squeeze=True)
        vec_squeezed_expected = np.array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])

        assert isinstance(vec_squeezed, np.ndarray)
        assert vec_squeezed.shape == (ny,)
        assert np.array_equal(vec_squeezed, vec_squeezed_expected)

        assert isinstance(vec, np.ndarray)
        assert vec.shape == (ny, 1)
        assert np.array_equal(vec, vec_squeezed_expected[:, np.newaxis])

    def test_copy_returns_petdataframe(self):
        '''Test that the copy method returns a PETDataFrame.'''
        pdf = PETDataFrame.from_pandas(self.simple_df)
        copy = pdf.copy()
        assert isinstance(copy, PETDataFrame)
    
    def test_loc_filtering_returns_petdataframe(self):
        '''Test that loc filtering returns a PETDataFrame.'''
        pdf = PETDataFrame.from_pandas(self.simple_df)
        subset = pdf.loc[['idx1']]
        assert isinstance(subset, PETDataFrame)
    
    def test_arithmetic_returns_petdataframe(self):
        '''Test that arithmetic operations return a PETDataFrame.'''
        pdf = PETDataFrame.from_pandas(self.simple_df)
        result = pdf + 1
        assert isinstance(result, PETDataFrame)



class TestMultiColumnJacobian:

    np.random.seed(404)
    data_dict = {
        ("keyA", "param1"): [np.random.rand(nx) for _ in range(nr)],
        ("keyA", "param2"): [np.random.rand(nx) for _ in range(nr)],
        ("keyA", "param3"): [np.random.rand(nx) for _ in range(nr)],
        ("keyB", "param1"): [np.random.rand(nx) for _ in range(nr)],
        ("keyB", "param2"): [np.random.rand(nx) for _ in range(nr)],
        ("keyB", "param3"): [np.random.rand(nx) for _ in range(nr)],
        ("keyC", "param1"): [np.random.rand(nx) for _ in range(nr)],
        ("keyC", "param2"): [np.random.rand(nx) for _ in range(nr)],
        ("keyC", "param3"): [np.random.rand(nx) for _ in range(nr)],
    }

    def test_to_series_multicolumn(self, multicolumn_dataframe):
        '''Test that the to_series method correctly converts a multi-column PETDataFrame to a pandas Series.'''
        pdf = PETDataFrame.from_pandas(multicolumn_dataframe)
        series = pdf.to_series()
        assert isinstance(series, pd.Series)
        assert series.shape == (nr*nc*nparams,)

    def test_to_matrix_multicolumn(self, multicolumn_dataframe):
        '''Test that the to_matrix method correctly converts a multi-column PETDataFrame to a numpy array.'''
        pdf = PETDataFrame.from_pandas(multicolumn_dataframe)
        matrix = pdf.to_matrix(is_jacobian=True)
        expected_rows = []
        keys = ("keyA", "keyB", "keyC")
        params = ("param1", "param2", "param3")
        for row_idx in range(nr):
            for key in keys:
                expected_rows.append(
                    np.concatenate([self.data_dict[(key, param)][row_idx] for param in params])
                )
        expected_matrix = np.stack(expected_rows)

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (ny, nx * nparams)
        assert np.array_equal(matrix, expected_matrix)


class TestEnsembleJacobian:

    def test_merge_ensemble_multicolumn(self, ensemble_dataframe):
        '''Test that the merge_ensemble method correctly merges a list of multi-column PETDataFrames into a single PETDataFrame.'''
        merged_pdf = PETDataFrame.merge_dataframes(ensemble_dataframe)
        assert isinstance(merged_pdf, PETDataFrame)
        assert merged_pdf.iloc[0]['keyA'].shape == (nx*nparams, ne)

    def test_to_matrix_ensemble_multicolumn(self, ensemble_dataframe):
        '''Test that the to_matrix method correctly converts a merged multi-column PETDataFrame to a numpy array.'''
        merged_pdf = PETDataFrame.merge_dataframes(ensemble_dataframe)
        matrix = merged_pdf.to_matrix(is_jacobian=True)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (ny, nx*nparams, ne)
    
    def test_to_matrix_multicolumn_ensemble(self, multicolumn_ensemble_dataframe, ensemble_dataframe):
        '''Test that the to_matrix method correctly converts a list of multi-column PETDataFrames to a numpy array.'''
        pdfs1 = PETDataFrame.merge_dataframes(multicolumn_ensemble_dataframe)
        pdfs2 = PETDataFrame.merge_dataframes(ensemble_dataframe)
        matrix = PETDataFrame.to_matrix(pdfs1, is_jacobian=True)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (ny, nx*nparams, ne)
        assert np.array_equal(matrix, PETDataFrame.to_matrix(pdfs2, is_jacobian=True))


class TestWithFieldData:

    data1 = {
        'keyScalar1': [1.0, 2.0, 3.0],
        'keyScalar2': [4.0, 5.0, 6.0],
        'keyField': [None, np.array([7.0, 8.0, 9.0, 10]), None]
    }
    data2 = {
        'keyScalar1': [10.0, 20.0, 30.0],
        'keyScalar2': [40.0, 50.0, 60.0],
        'keyField': [None, np.array([70.0, 80.0, 90.0, 100]), None]
    }
    pdf1 = PETDataFrame(data=data1, index=['idx1', 'idx2', 'idx3'])
    pdf2 = PETDataFrame(data=data2, index=['idx1', 'idx2', 'idx3'])

    def test_to_matrix_with_field(self):
        '''Test that the to_matrix method correctly handles a PETDataFrame with a field column.'''
        vec_filtered = self.pdf1.to_matrix(filter=True, is_jacobian=False, squeeze=True)
        vec_filtered_expected = np.array([1.0, 4.0, 2.0, 5.0, 7.0, 8.0, 9.0, 10.0, 3.0, 6.0])

        vec_unfiltered = self.pdf1.to_matrix(filter=False, is_jacobian=False, squeeze=True)
        vec_unfiltered_expected = np.array([1.0, 4.0, None, 2.0, 5.0, 7.0, 8.0, 9.0, 10.0, 3.0, 6.0, None])

        assert isinstance(vec_filtered, np.ndarray)
        assert vec_filtered.shape == (10,)
        assert np.array_equal(vec_filtered, vec_filtered_expected)

        assert isinstance(vec_unfiltered, np.ndarray)
        assert vec_unfiltered.shape == (12,)
        assert np.array_equal(vec_unfiltered, vec_unfiltered_expected)

    def test_to_ensemble_matrix_with_field(self):
        '''Test that the to_matrix method correctly handles a list of PETDataFrames with a field column.'''
        merged_pdf = PETDataFrame.merge_dataframes([self.pdf1, self.pdf2])
        matrix_filtered = merged_pdf.to_matrix(filter=True, is_jacobian=False)
        matrix_filtered_expected = np.array([
            [1.0, 10.0],
            [4.0, 40.0],
            [2.0, 20.0],
            [5.0, 50.0],
            [7.0, 70.0],
            [8.0, 80.0],
            [9.0, 90.0],
            [10.0, 100.0],
            [3.0, 30.0],
            [6.0, 60.0]
        ])

        matrix_unfiltered = merged_pdf.to_matrix(filter=False, is_jacobian=False)
        matrix_unfiltered_expected = np.array([
            [1.0, 10.0],
            [4.0, 40.0],
            [None, None],
            [2.0, 20.0],
            [5.0, 50.0],  
            [7.0, 70.0],
            [8.0, 80.0],
            [9.0, 90.0],
            [10.0, 100.0],
            [3.0, 30.0],
            [6.0, 60.0],
            [None, None]
        ])

        assert isinstance(matrix_filtered, np.ndarray)
        assert matrix_filtered.shape == (10, 2)
        assert np.array_equal(matrix_filtered, matrix_filtered_expected)

        assert isinstance(matrix_unfiltered, np.ndarray)
        assert matrix_unfiltered.shape == (12, 2)
        assert np.array_equal(matrix_unfiltered, matrix_unfiltered_expected)
    




# ==============================================================================
# PETStateArray TESTS
# ==============================================================================

@pytest.fixture
def sample_state_array():
    nstate = nx * nparams
    # Start from 1.0 to avoid division-by-zero in operator tests
    data = np.arange(1, nstate * ne + 1, dtype=float).reshape(nstate, ne)
    indices = {
        f'key{p+1}': (p * nx, (p + 1) * nx)
        for p in range(nparams)
    }
    return PETStateArray(data, indices=indices)

class TestPETStateArray:

    def test_construct_from_ndarray(self, sample_state_array):
        '''Test that PETStateArray can be created from a numpy array.'''
        assert isinstance(sample_state_array, PETStateArray)
        assert sample_state_array.shape == (nx * nparams, ne)
        assert len(sample_state_array.indices) == nparams
        assert sample_state_array.state_axis == 0

    def test_to_dict_shapes(self, sample_state_array):
        '''Test that to_dict returns a dict with correct shapes per key.'''
        state_dict = sample_state_array.to_dict()
        assert isinstance(state_dict, dict)
        assert len(state_dict) == nparams
        for val in state_dict.values():
            assert val.shape == (nx, ne)

    def test_from_dict(self):
        '''Test that from_dict reconstructs a PETStateArray with correct shape and indices.'''
        member = {f'key{p+1}': np.random.randn(nx, ne) for p in range(nparams)}
        state = PETStateArray.from_dict(member, ne=ne)
        assert isinstance(state, PETStateArray)
        assert state.shape == (nx * nparams, ne)
        assert list(state.indices.keys()) == [f'key{p+1}' for p in range(nparams)]

    def test_to_list_of_dicts_roundtrip(self, sample_state_array):
        '''Test that to_list_of_dicts / from_list_of_dicts is a lossless roundtrip.'''
        members = sample_state_array.to_list_of_dicts()
        rebuilt = PETStateArray.from_list_of_dicts(members)
        assert rebuilt.shape == sample_state_array.shape
        assert np.allclose(np.asarray(rebuilt), np.asarray(sample_state_array))

    def test_transpose_flips_state_axis(self, sample_state_array):
        '''Test that .T flips state_axis while preserving indices.'''
        transposed = sample_state_array.T
        assert isinstance(transposed, PETStateArray)
        assert transposed.shape == (ne, nx * nparams)
        assert transposed.indices == sample_state_array.indices
        assert transposed.state_axis == 1

    def test_is_numpy_subclass(self, sample_state_array):
        '''PETStateArray must be a numpy ndarray subclass.'''
        assert isinstance(sample_state_array, np.ndarray)
        assert isinstance(sample_state_array, PETStateArray)


class TestPETStateArrayOperators:
    '''
    Test that every operator defined on PETStateArray returns a PETStateArray
    with the correct values, indices, and state_axis preserved.
    '''

    def _check(self, result, reference, expected_values):
        assert isinstance(result, PETStateArray)
        assert result.indices == reference.indices
        assert result.state_axis == reference.state_axis
        assert np.allclose(np.asarray(result), expected_values)

    # --- scalar binary operators ---

    def test_add(self, sample_state_array):
        '''a + scalar'''
        a = np.asarray(sample_state_array)
        self._check(sample_state_array + 5.0, sample_state_array, a + 5.0)

    def test_radd(self, sample_state_array):
        '''scalar + a'''
        a = np.asarray(sample_state_array)
        self._check(5.0 + sample_state_array, sample_state_array, 5.0 + a)

    def test_sub(self, sample_state_array):
        '''a - scalar'''
        a = np.asarray(sample_state_array)
        self._check(sample_state_array - 3.0, sample_state_array, a - 3.0)

    def test_rsub(self, sample_state_array):
        '''scalar - a'''
        a = np.asarray(sample_state_array)
        self._check(1000.0 - sample_state_array, sample_state_array, 1000.0 - a)

    def test_mul(self, sample_state_array):
        '''a * scalar'''
        a = np.asarray(sample_state_array)
        self._check(sample_state_array * 2.0, sample_state_array, a * 2.0)

    def test_rmul(self, sample_state_array):
        '''scalar * a'''
        a = np.asarray(sample_state_array)
        self._check(2.0 * sample_state_array, sample_state_array, 2.0 * a)

    def test_truediv(self, sample_state_array):
        '''a / scalar'''
        a = np.asarray(sample_state_array)
        self._check(sample_state_array / 2.0, sample_state_array, a / 2.0)

    def test_rtruediv(self, sample_state_array):
        '''scalar / a'''
        a = np.asarray(sample_state_array)
        self._check(1000.0 / sample_state_array, sample_state_array, 1000.0 / a)

    def test_floordiv(self, sample_state_array):
        '''a // scalar'''
        a = np.asarray(sample_state_array)
        self._check(sample_state_array // 3.0, sample_state_array, a // 3.0)

    def test_pow(self, sample_state_array):
        '''a ** scalar'''
        a = np.asarray(sample_state_array)
        self._check(sample_state_array ** 2.0, sample_state_array, a ** 2.0)

    # --- array binary operators ---

    def test_add_array(self, sample_state_array):
        '''a + b where b is a plain ndarray of the same shape'''
        b = np.ones_like(sample_state_array)
        a = np.asarray(sample_state_array)
        self._check(sample_state_array + b, sample_state_array, a + b)

    def test_sub_array(self, sample_state_array):
        '''a - b'''
        b = np.ones_like(sample_state_array)
        a = np.asarray(sample_state_array)
        self._check(sample_state_array - b, sample_state_array, a - b)

    def test_mul_array(self, sample_state_array):
        '''a * b element-wise'''
        b = np.full_like(sample_state_array, 2.0)
        a = np.asarray(sample_state_array)
        self._check(sample_state_array * b, sample_state_array, a * b)

    # --- matmul ---

    def test_matmul(self, sample_state_array):
        '''a @ M where M transforms ensemble axis'''
        M = np.eye(ne)  # identity: result == a
        a = np.asarray(sample_state_array)
        result = sample_state_array @ M
        assert isinstance(result, PETStateArray)
        assert np.allclose(np.asarray(result), a @ M)

    def test_rmatmul(self, sample_state_array):
        '''M @ a'''
        M = np.eye(nx * nparams)
        a = np.asarray(sample_state_array)
        result = M @ sample_state_array
        assert isinstance(result, PETStateArray)
        assert np.allclose(np.asarray(result), M @ a)

    # --- unary operators ---

    def test_neg(self, sample_state_array):
        '''-a'''
        a = np.asarray(sample_state_array)
        self._check(-sample_state_array, sample_state_array, -a)

    def test_pos(self, sample_state_array):
        '''+a'''
        a = np.asarray(sample_state_array)
        self._check(+sample_state_array, sample_state_array, +a)

    def test_abs(self, sample_state_array):
        '''abs(-a) == a (all elements are positive)'''
        a = np.asarray(sample_state_array)
        self._check(abs(-sample_state_array), sample_state_array, np.abs(-a))

    # --- chained expression ---

    def test_operator_chain(self, sample_state_array):
        '''(a + 1) * 2 - 0.5 remains a PETStateArray with correct values'''
        a = np.asarray(sample_state_array)
        result = (sample_state_array + 1.0) * 2.0 - 0.5
        self._check(result, sample_state_array, (a + 1.0) * 2.0 - 0.5)


