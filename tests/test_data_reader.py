import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from misc.structures import PETDataFrame
from misc.read_input_csv import DataReader

@pytest.fixture
def simple_data_df():
    d = {
        'keyA': [1.0, 2.0],
        'keyB': [3.0, 4.0],
        'keyC': [5.0, 6.0]
    }
    index = ['idx1', 'idx2']
    index_name = 'index'
    data_df = pd.DataFrame(d, index=index)
    data_df.index.name = index_name
    return data_df

@pytest.fixture
def simple_data_var_df_abs(simple_data_df):
    var_d = {
        'keyA': [['abs', 0.1], 
                 ['abs', 0.2]],

        'keyB': [['abs', 0.3], 
                 ['abs', 0.4]],

        'keyC': [['abs', 0.5], 
                 ['abs', 0.6]]
    }
    var_df = pd.DataFrame(var_d, index=simple_data_df.index)
    var_df.index.name = simple_data_df.index.name
    return var_df

@pytest.fixture
def simple_data_var_df_rel(simple_data_df):
    sdf = simple_data_df
    var_d = {
        'keyA': [
            ['rel', float(np.sqrt(0.1) / (sdf.loc['idx1', 'keyA'] * 0.01))],
            ['rel', float(np.sqrt(0.2) / (sdf.loc['idx2', 'keyA'] * 0.01))],
        ],
        'keyB': [
            ['rel', float(np.sqrt(0.3) / (sdf.loc['idx1', 'keyB'] * 0.01))],
            ['rel', float(np.sqrt(0.4) / (sdf.loc['idx2', 'keyB'] * 0.01))],
        ],
        'keyC': [
            ['rel', float(np.sqrt(0.5) / (sdf.loc['idx1', 'keyC'] * 0.01))],
            ['rel', float(np.sqrt(0.6) / (sdf.loc['idx2', 'keyC'] * 0.01))],
        ],
    }   
    var_df = pd.DataFrame(var_d, index=sdf.index)
    var_df.index.name = sdf.index.name
    return var_df


@pytest.fixture
def df_with_npz():  
    # Create a simple DataFrame and save it as .npz
    d = {
        'keyA': [1.0, 2.0, 3.0],
        'keyB': [4.0, 5.0, 6.0],
        'keyC': [7.0, 8.0, 9.0],
        'keyNPZ': [None, 'file.npz', None]
    }
    index = ['idx1', 'idx2', 'idx3']
    index_name = 'index'
    data_df = pd.DataFrame(d, index=index)
    data_df.index.name = index_name    
    return data_df

class TestSimpleData:

    def test_read_pickle(self, tmp_path, simple_data_df):
        pickle_path = tmp_path / "test_data.pkl"
        simple_data_df.to_pickle(pickle_path)
        
        # Use DataReader to read the pickle file
        reader = DataReader({'data': str(pickle_path), 'datavar': ''})
        data_df = reader.get_data()

        assert isinstance(data_df, PETDataFrame)
        assert data_df.equals(simple_data_df)
        assert data_df.columns.equals(simple_data_df.columns)
        assert data_df.index.equals(simple_data_df.index)
        assert data_df.index.name == simple_data_df.index.name

    def test_read_csv(self, tmp_path, simple_data_df):
        csv_path = tmp_path / "test_data.csv"
        simple_data_df.to_csv(csv_path)
        
        # Use DataReader to read the CSV file
        reader = DataReader({'data': str(csv_path), 'datavar': ''})
        data_df = reader.get_data()
        
        assert isinstance(data_df, PETDataFrame)
        assert data_df.equals(simple_data_df)
        assert data_df.columns.equals(simple_data_df.columns)
        assert data_df.index.equals(simple_data_df.index)
        assert data_df.index.name == simple_data_df.index.name


class TestSimpleDataVariance:

    d = {
        'keyA': [0.1, 0.2],
        'keyB': [0.3, 0.4],
        'keyC': [0.5, 0.6]
    }
    true_var_df = PETDataFrame(d, index=['idx1', 'idx2'])
    true_var_df.index.name = 'index'

    def test_read_abs_variance_pickle(self, tmp_path, simple_data_df, simple_data_var_df_abs):
        pickle_path_data = tmp_path / "test_data.pkl"
        simple_data_df.to_pickle(pickle_path_data)

        pickle_path_var = tmp_path / "test_data_var.pkl"
        simple_data_var_df_abs.to_pickle(pickle_path_var)
        
        # Use DataReader to read the pickle file
        options = {
            'data': str(pickle_path_data),
            'datavar': str(pickle_path_var)
        }
        reader = DataReader(options)
        data_df = reader.get_data() 
        data_var_df = reader.get_variance(data_df)
        
        assert isinstance(data_var_df, PETDataFrame)
        assert data_var_df.equals(self.true_var_df)
        assert data_var_df.columns.equals(self.true_var_df.columns)
        assert data_var_df.index.equals(self.true_var_df.index)
        assert data_var_df.index.name == self.true_var_df.index.name
    
    
    def test_read_rel_variance_pickle(self, tmp_path, simple_data_df, simple_data_var_df_rel):
        pickle_path_data = tmp_path / "test_data.pkl"
        simple_data_df.to_pickle(pickle_path_data)

        pickle_path_var = tmp_path / "test_data_var.pkl"
        simple_data_var_df_rel.to_pickle(pickle_path_var)
        
        # Use DataReader to read the pickle file
        options = {
            'data': str(pickle_path_data),
            'datavar': str(pickle_path_var)
        }
        reader = DataReader(options)
        data_df = reader.get_data() 
        data_var_df = reader.get_variance(data_df)

        assert isinstance(data_var_df, PETDataFrame)
        assert_frame_equal(data_var_df, self.true_var_df, atol=1e-12, rtol=1e-12)
        assert data_var_df.columns.equals(self.true_var_df.columns)
        assert data_var_df.index.equals(self.true_var_df.index)
        assert data_var_df.index.name == self.true_var_df.index.name


    def test_read_abs_variance_csv(self, tmp_path, simple_data_df, simple_data_var_df_abs):
        csv_path_data = tmp_path / "test_data.csv"
        simple_data_df.to_csv(csv_path_data)

        csv_path_var = tmp_path / "test_data_var.csv"
        simple_data_var_df_abs.to_csv(csv_path_var)
        
        # Use DataReader to read the CSV file
        options = {
            'data': str(csv_path_data),
            'datavar': str(csv_path_var)
        }
        reader = DataReader(options)
        data_df = reader.get_data() 
        data_var_df = reader.get_variance(data_df)
        
        assert isinstance(data_var_df, PETDataFrame)
        assert_frame_equal(data_var_df, self.true_var_df, atol=1e-12, rtol=1e-12)
        assert data_var_df.columns.equals(self.true_var_df.columns)
        assert data_var_df.index.equals(self.true_var_df.index)
        assert data_var_df.index.name == self.true_var_df.index.name


class TestDataReaderWihtNPZ:

    def test_read_data_with_npz(self, tmp_path, df_with_npz):
        # Create a temporary .npz file with the DataFrame
        npz_path = tmp_path / "file.npz"
        array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        np.savez(npz_path, array)

        # Use absolute path so DataReader can resolve the file from any cwd
        df_with_npz.loc['idx2', 'keyNPZ'] = str(npz_path)

        # create a temporary pkl file 
        pkl_path = tmp_path / "test_data.pkl"
        df_with_npz.to_pickle(pkl_path)

        # Use DataReader to read the CSV file that references the .npz file
        reader = DataReader({'data': str(pkl_path), 'datavar': ''})
        data_df = reader.get_data()

        vec = np.array(
            [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 
             10, 20, 30, 40, 50, 60, 70, 
             80, 90, 100, 3.0, 6.0, 9.0]
        )

        assert isinstance(data_df, PETDataFrame)
        assert data_df.loc['idx2', 'keyNPZ'].shape == (10,)
        assert np.array_equal(data_df.loc['idx2', 'keyNPZ'], array)
        assert data_df.to_matrix().shape == (len(array) + 9,)
        assert np.array_equal(data_df.to_matrix(), vec)


# Need Help Here!!
class TestDataReaderWithCompression:
    pass
        
        


