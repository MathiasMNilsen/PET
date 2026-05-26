"""
Comprehensive tests for PETDataFrame and PETStateArray.

This suite preserves:
- Exact numerical correctness
- Deterministic behavior
- Full operator coverage
- Field data edge cases
- Scaling consistency
"""

import numpy as np
import pandas as pd
import pytest

from misc.structures.structures import PETDataFrame, PETStateArray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NPARAMS = 3
NX = 8
NROWS = 2
NCOLS = 3
NY = NROWS * NCOLS
NE = 10

INDEX = ["idx1", "idx2"]
INDEX_NAME = "index"


# ---------------------------------------------------------------------------
# Deterministic MultiIndex Data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multicolumn_ensemble():
    """Generate deterministic ensemble of multi-column DataFrames."""
    np.random.seed(404)

    dfs = []
    for _ in range(NE):
        data = {}
        for key in ("keyA", "keyB", "keyC"):
            for param in ("param1", "param2", "param3"):
                data[(key, param)] = [
                    np.random.rand(NX) for _ in range(NROWS)
                ]

        df = pd.DataFrame(data, index=INDEX)
        df.columns = pd.MultiIndex.from_tuples(data.keys())
        df.index.name = INDEX_NAME
        dfs.append(df)

    return dfs


@pytest.fixture
def multicolumn_df(multicolumn_ensemble):
    return multicolumn_ensemble[0]


@pytest.fixture
def ensemble_singlelevel(multicolumn_ensemble):
    return [
        PETDataFrame._to_singlelevel_columns(df)
        for df in multicolumn_ensemble
    ]


# ---------------------------------------------------------------------------
# PETDataFrame: Basic
# ---------------------------------------------------------------------------

class TestPETDataFrameBasic:

    def setup_method(self):
        self.data = {
            "keyA": [1.0, 2.0],
            "keyB": [3.0, 4.0],
            "keyC": [5.0, 6.0],
        }

        self.df = pd.DataFrame(self.data, index=INDEX)
        self.df.index.name = INDEX_NAME
        self.df.attrs["units"] = {
            k: f"unit:{k}" for k in self.data
        }

    def test_from_pandas(self):
        pdf = PETDataFrame.from_pandas(self.df)

        expected = PETDataFrame(self.data, index=INDEX)
        expected.index.name = INDEX_NAME

        assert pdf.equals(expected)

    def test_attrs_preserved(self):
        pdf = PETDataFrame.from_pandas(self.df)
        assert pdf.attrs["units"] == self.df.attrs["units"]

    def test_to_matrix(self):
        pdf = PETDataFrame.from_pandas(self.df)

        vec = pdf.to_matrix(squeeze=False)
        vec_sq = pdf.to_matrix(squeeze=True)

        expected = np.array([1, 3, 5, 2, 4, 6], dtype=float)

        assert vec.shape == (NY, 1)
        assert np.array_equal(vec[:, 0], expected)

        assert vec_sq.shape == (NY,)
        assert np.array_equal(vec_sq, expected)

    def test_return_types(self):
        pdf = PETDataFrame.from_pandas(self.df)

        assert isinstance(pdf.copy(), PETDataFrame)
        assert isinstance(pdf.loc[["idx1"]], PETDataFrame)
        assert isinstance(pdf + 1, PETDataFrame)


# ---------------------------------------------------------------------------
# Jacobian (Multi-column)
# ---------------------------------------------------------------------------

class TestMultiColumnJacobian:

    def test_to_series(self, multicolumn_df):
        pdf = PETDataFrame.from_pandas(multicolumn_df)
        series = pdf.to_series()

        assert isinstance(series, pd.Series)
        assert series.shape == (NROWS * NCOLS * NPARAMS,)

    def test_to_matrix_exact(self, multicolumn_df):
        pdf = PETDataFrame.from_pandas(multicolumn_df)
        matrix = pdf.to_matrix(is_jacobian=True)

        expected_rows = []
        keys = ("keyA", "keyB", "keyC")
        params = ("param1", "param2", "param3")

        for r in range(NROWS):
            for key in keys:
                row = np.concatenate([
                    multicolumn_df[(key, param)][r]
                    for param in params
                ])
                expected_rows.append(row)

        expected = np.stack(expected_rows)

        assert matrix.shape == (NY, NX * NPARAMS)
        assert np.array_equal(matrix, expected)


# ---------------------------------------------------------------------------
# Ensemble handling
# ---------------------------------------------------------------------------

class TestEnsembleJacobian:

    def test_merge(self, ensemble_singlelevel):
        merged = PETDataFrame.merge_dataframes(ensemble_singlelevel)

        assert merged.iloc[0]["keyA"].shape == (NX * NPARAMS, NE)

    def test_matrix_shape(self, ensemble_singlelevel):
        merged = PETDataFrame.merge_dataframes(ensemble_singlelevel)
        matrix = merged.to_matrix(is_jacobian=True)

        assert matrix.shape == (NY, NX * NPARAMS, NE)

    def test_multi_vs_single_consistency(
        self, multicolumn_ensemble, ensemble_singlelevel
    ):
        merged_multi = PETDataFrame.merge_dataframes(multicolumn_ensemble)
        merged_single = PETDataFrame.merge_dataframes(ensemble_singlelevel)

        mat1 = PETDataFrame.to_matrix(merged_multi, is_jacobian=True)
        mat2 = PETDataFrame.to_matrix(merged_single, is_jacobian=True)

        assert np.array_equal(mat1, mat2)


# ---------------------------------------------------------------------------
# Field data
# ---------------------------------------------------------------------------

class TestFieldData:

    def setup_method(self):
        self.pdf1 = PETDataFrame(
            {
                "keyScalar1": [1, 2, 3],
                "keyScalar2": [4, 5, 6],
                "keyField": [None, np.array([7, 8, 9, 10]), None],
            },
            index=["idx1", "idx2", "idx3"],
        )

        self.pdf2 = PETDataFrame(
            {
                "keyScalar1": [10, 20, 30],
                "keyScalar2": [40, 50, 60],
                "keyField": [None, np.array([70, 80, 90, 100]), None],
            },
            index=["idx1", "idx2", "idx3"],
        )

    def test_filtered_unfiltered_vectors(self):
        vec_f = self.pdf1.to_matrix(filter=True, squeeze=True)
        vec_u = self.pdf1.to_matrix(filter=False, squeeze=True)

        expected_f = np.array([1,4,2,5,7,8,9,10,3,6], dtype=float)
        expected_u = np.array(
            [1,4,None,2,5,7,8,9,10,3,6,None], dtype=object
        )

        assert np.array_equal(vec_f, expected_f)
        assert np.array_equal(vec_u, expected_u)

    def test_ensemble_matrix(self):
        merged = PETDataFrame.merge_dataframes([self.pdf1, self.pdf2])

        mat_f = merged.to_matrix(filter=True)
        mat_u = merged.to_matrix(filter=False)

        expected_f = np.array([
            [1,10],[4,40],[2,20],[5,50],
            [7,70],[8,80],[9,90],[10,100],
            [3,30],[6,60]
        ])

        expected_u = np.array([
            [1,10],[4,40],[None,None],[2,20],[5,50],
            [7,70],[8,80],[9,90],[10,100],
            [3,30],[6,60],[None,None]
        ], dtype=object)

        assert np.array_equal(mat_f, expected_f)
        assert np.array_equal(mat_u, expected_u)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

class TestScaling:

    def setup_method(self):
        np.random.seed(404)

        self.data = PETDataFrame(
            {k: 10*np.random.rand(5) for k in ("keyA","keyB","keyC")}
        )
        self.var = PETDataFrame(
            {k: 0.1*np.random.rand(5) for k in ("keyA","keyB","keyC")}
        )
        self.jac = PETDataFrame(
            {
                (k,"param1"): [50*np.random.rand(NX,NE) for _ in range(5)]
                for k in ("keyA","keyB","keyC")
            }
        )

    def test_max_min(self):
        scaled = self.data.copy()
        scaled.scale(type="max-min")

        inv = scaled.copy()
        inv.invert_scale(type="max-min")

        assert scaled.is_scaled
        assert np.all((scaled >= 0) & (scaled <= 1))
        pd.testing.assert_frame_equal(inv, self.data)

    def test_variance(self):
        scaled = self.data.copy()
        scaled.scale(type="max-min")

        rng = scaled.scale_max - scaled.scale_min

        expected = self.var / (rng**2)

        var_scaled = self.var.copy()
        var_scaled.scale(type="max-min", minimum=0, maximum=rng**2)

        inv = var_scaled.copy()
        inv.invert_scale(type="max-min")

        pd.testing.assert_frame_equal(var_scaled, expected)
        pd.testing.assert_frame_equal(inv, self.var)

    def test_jacobian(self):
        scaled = self.data.copy()
        scaled.scale(type="max-min")

        rng = scaled.scale_max - scaled.scale_min

        expected = self.jac.div(rng, axis="columns", level=0)

        jac_scaled = self.jac.copy()
        jac_scaled.scale(type="max-min", minimum=0, maximum=rng)

        inv = jac_scaled.copy()
        inv.invert_scale(type="max-min")

        pd.testing.assert_frame_equal(jac_scaled, expected)
        pd.testing.assert_frame_equal(inv, self.jac)


# ---------------------------------------------------------------------------
# PETStateArray
# ---------------------------------------------------------------------------

@pytest.fixture
def state_array():
    data = np.arange(1, NX * NPARAMS * NE + 1, dtype=float)
    data = data.reshape(NX * NPARAMS, NE)

    indices = {
        f"key{i+1}": (i*NX, (i+1)*NX)
        for i in range(NPARAMS)
    }

    return PETStateArray(data, indices=indices)


# ---------------------------------------------------------------------------
# PETStateArray: Basic
# ---------------------------------------------------------------------------

class TestStateArrayBasic:

    def test_shapes(self, state_array):
        assert state_array.shape == (NX * NPARAMS, NE)

    def test_dict_conversion(self, state_array):
        d = state_array.to_dict()
        assert all(v.shape == (NX, NE) for v in d.values())

    def test_roundtrip(self, state_array):
        rebuilt = PETStateArray.from_list_of_dicts(
            state_array.to_list_of_dicts()
        )
        assert np.allclose(rebuilt, state_array)

    def test_transpose(self, state_array):
        t = state_array.T
        assert t.shape == (NE, NX * NPARAMS)
        assert t.indices == state_array.indices
        assert t.state_axis == 1


# ---------------------------------------------------------------------------
# PETStateArray Operators (FULL COVERAGE)
# ---------------------------------------------------------------------------

class TestStateArrayOperators:

    def _check(self, result, ref, expected):
        assert isinstance(result, PETStateArray)
        assert result.indices == ref.indices
        assert result.state_axis == ref.state_axis
        assert np.allclose(result, expected)

    def test_all_ops(self, state_array):
        a = np.asarray(state_array)
        b = np.ones_like(a) * 2

        # scalar ops
        self._check(state_array + 5, state_array, a + 5)
        self._check(5 + state_array, state_array, 5 + a)
        self._check(state_array - 3, state_array, a - 3)
        self._check(1000 - state_array, state_array, 1000 - a)
        self._check(state_array * 2, state_array, a * 2)
        self._check(2 * state_array, state_array, 2 * a)
        self._check(state_array / 2, state_array, a / 2)
        self._check(1000 / state_array, state_array, 1000 / a)
        self._check(state_array // 3, state_array, a // 3)
        self._check(state_array ** 2, state_array, a ** 2)

        # array ops
        self._check(state_array + b, state_array, a + b)
        self._check(state_array - b, state_array, a - b)
        self._check(state_array * b, state_array, a * b)

        # unary
        self._check(-state_array, state_array, -a)
        self._check(+state_array, state_array, +a)
        self._check(abs(state_array), state_array, np.abs(a))

        # chained
        self._check(
            (state_array + 1) * 2 - 0.5,
            state_array,
            (a + 1) * 2 - 0.5,
        )


