"""
CSV and Pickle Data Reader Utilities

This module provides utility functions for reading and processing data from CSV and pickle files.
It supports various data formats including NumPy arrays, pandas DataFrames, and handles data
type conversions for ensemble modeling and data assimilation workflows.

Main Functions:
    - read_data_df: Reads data from CSV/pickle files, returns as NumPy arrays or dictionaries
    - read_var_df: Reads variance data from CSV/pickle files
    - read_data_csv: Legacy CSV reading function with data flattening
    - read_var_csv: Legacy variance CSV reading function
    - convert_to_array: Converts string representations to NumPy arrays
    - to_array_if_sequence: Converts various data types to NumPy array format

Typical use cases:
    - Loading observational data for data assimilation
    - Reading ensemble data with various data types
    - Processing CSV files with mixed data types and array-like strings
    - Handling variance/uncertainty data alongside measurements

Last Modified: February 2026
"""

import ast
import pandas as pd
import numpy as np
import pickle

def convert_to_array(array_str):
    """
    Convert space-separated string representations of numbers to NumPy arrays.
    
    This function handles strings with space-separated numeric values and converts
    them back to NumPy arrays. It removes brackets and whitespace before parsing.
    
    Parameters
    ----------
    array_str : str
        String containing space-separated numbers, optionally with brackets.
        Example: "[1.0 2.0 3.0]" or "1.0 2.0 3.0"
    
    Returns
    -------
    np.ndarray or str
        NumPy array of floats if conversion is successful, otherwise returns
        the original string unchanged.
    
    Examples
    --------
    >>> convert_to_array("1.0 2.0 3.0")
    array([1., 2., 3.])
    >>> convert_to_array("[1.0 2.0 3.0]")
    array([1., 2., 3.])
    """
    try:
        # Remove any unwanted characters like square brackets and split by space
        cleaned_str = array_str.replace('[', '').replace(']', '').strip()
        # Split the string by spaces and convert the result to a NumPy array of floats
        return np.array([float(x) for x in cleaned_str.split()])
    except (ValueError, AttributeError):
        # If the string cannot be converted, return it as is (error handling)
        return array_str

def to_array_if_sequence(val):
    """
    Convert various data types to NumPy array or sequence format.
    
    Handles conversion of different input types (scalars, lists, strings, arrays)
    into a consistent array-like format for data processing.
    
    Parameters
    ----------
    val : various
        Input value to convert. Can be np.ndarray, int, float, list, str, or other.
    
    Returns
    -------
    np.ndarray or list
        - NumPy array if input is ndarray, numeric scalar, list, or parseable string
        - List containing the value if input is of another type
    
    Notes
    -----
    String inputs are only parsed if they are enclosed in brackets (e.g., "[1 2 3]").
    All numeric scalars are wrapped into 1D arrays.
    """
    if isinstance(val, np.ndarray):
        return val
    elif isinstance(val, (int, float)):
        return np.array([val])
    elif isinstance(val, list):
        return np.array(val)
    elif isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
        try:
            return np.fromstring(val.strip('[]'), sep=' ')
        except:
            return val  # fallback in case parsing fails
    else:
        return [val]  # wrap scalars


def read_data_df(filename, datatype=None, truedataindex=None, outtype='np.array',return_data_info=True):
    """
    Read observational data from CSV or pickle files with flexible output formats.
    
    This function reads data files (CSV or pickle) containing observational data,
    processes array-like string representations, and returns the data in the
    requested format. Supports filtering by data types and row indices.
    
    Parameters
    ----------
    filename : str
        Path to the data file. Must end with '.csv' or '.pkl'.
    datatype : list of str, optional
        Column names to extract. If None, all columns are used. Default is None.
    truedataindex : list of int, optional
        Row indices to extract (0-based). If None, all rows are used. Default is None.
    outtype : {'np.array', 'list'}, optional
        Output format:
        - 'np.array': Returns flattened NumPy array
        - 'list': Returns list of dictionaries
        Default is 'np.array'.
    return_data_info : bool, optional
        If True, also returns metadata (column names and row indices). Default is True.
    
    Returns
    -------
    flat_array : np.ndarray
        Flattened 1D array of all data (if outtype='np.array').
    data : list of dict
        List where each element is a dictionary with column names as keys (if outtype='list').
    datatype : list of str
        Column names used (only if return_data_info=True).
    indices : list
        Row indices/labels used (only if return_data_info=True).
    
    Notes
    -----
    - String representations of arrays (e.g., "[1.0 2.0 3.0]") are automatically
      converted to NumPy arrays.
    - When outtype='np.array', arrays from multiple columns and rows are concatenated
      into a single flat array.
    - The first column in CSV files is used as the index.
    """

    # read the file
    if filename.endswith('.csv'):
        df = pd.read_csv(filename, index_col=0)
    elif filename.endswith('.pkl'):
        df = pd.read_pickle(filename)
    # convert the string representation of arrays back to NumPy arrays
    for col in df.columns:
        df[col] = df[col].apply(convert_to_array)

    df = df.where(pd.notnull(df), None)

    if outtype == 'np.array': # vectorize data
        if datatype is not None:
            if truedataindex is not None:
                flat_array = np.concatenate([np.concatenate([df.iloc[ti][col] if isinstance(df.iloc[ti][col], np.ndarray) else
                                                             np.array([df.iloc[ti][col]])
                                                            for col in datatype]) for ti in truedataindex])
                if return_data_info:
                   return flat_array, list(datatype), [df.index[el] for el in truedataindex]
            else:
                flat_array = np.concatenate([np.concatenate([row[col] if isinstance(row[col], np.ndarray) else
                                                             np.array([row[col]])
                                                for col in datatype]) for _, row in df.iterrows()])
                if return_data_info:
                   return flat_array, list(datatype), list(df.index)
        else:
            if truedataindex is not None:
                flat_array = np.concatenate([np.concatenate([df.iloc[ti][col] if isinstance(df.iloc[ti][col], np.ndarray) else
                                                             np.array([df.iloc[ti][col]])
                                                            for col in df.columns]) for ti in truedataindex])
                if return_data_info:
                    return flat_array, list(df.columns), [df.index[el] for el in truedataindex]
            else:
                flat_array = np.concatenate([np.concatenate([row[col] if isinstance(row[col], np.ndarray) else np.array([row[col]])
                                for col in df.columns]) for _, row in df.iterrows()])
                if return_data_info:
                    return flat_array, list(df.columns), list(df.index)

        return flat_array

    elif outtype == 'list': # return data as a list over row indices. Where each list element is a dictionary with keys equal to column names
        if datatype is not None:
            if truedataindex is not None:
                data = [
                    {
                        col: to_array_if_sequence(df.iloc[ti][col])
                        for col in datatype
                    }
                    for ti in truedataindex
                ]
                
                if return_data_info:
                    data, list(datatype), [df.index[el] for el in truedataindex]
            else:
                data = [
                    {
                        col: to_array_if_sequence(row[col])
                        for col in datatype
                    }
                    for _, row in df.iterrows()
                ]
                if return_data_info:
                    data, list(datatype), list(df.index)
        else:
            if truedataindex is not None:
                data = [
                    {
                        col: to_array_if_sequence(df.iloc[ti][col])
                        for col in df.columns
                    }
                    for ti in truedataindex
                ]
                if return_data_info:
                    data, list(datatype), list(df.index)
            else:
                data = [
                    {
                        col: to_array_if_sequence(row[col])
                        for col in df.columns
                    }
                    for _, row in df.iterrows()
                ]
                if return_data_info:
                    return data, list(df.columns), list(df.index)
        return data

def read_var_df(filename, datatype=None, truedataindex=None, outtype='list'):
    """
    Read variance/uncertainty data from CSV or pickle files.
    
    This function is designed to read variance or standard deviation data that
    corresponds to observational data. It returns the data as a list of dictionaries,
    with special handling for datatype columns that may contain tuple representations.
    
    Parameters
    ----------
    filename : str
        Path to the variance file. Must end with '.csv' or '.pkl'.
    datatype : list of str, optional
        Column names to extract. Supports tuple-like string representations
        (e.g., "('OPR', 'WWCT')") which are parsed using ast.literal_eval.
        If None, all columns are used. Default is None.
    truedataindex : list of str or int, optional
        Row indices/labels to extract. If None, all rows are used. Default is None.
    outtype : {'list'}, optional
        Output format. Currently only 'list' is supported. Default is 'list'.
    
    Returns
    -------
    var : list of dict
        List where each element is a dictionary with column names as keys and
        variance/uncertainty values as values. Each dictionary corresponds to one row.
    
    Notes
    -----
    - CSV file indices are converted to strings for consistent lookup.
    - The datatype parameter attempts to evaluate string representations of tuples,
      which is useful when column names are composite keys.
    - This function is typically used alongside read_data_df to load both
      observations and their uncertainties.
    """

    # read the file
    if filename.endswith('.csv'):
        df = pd.read_csv(filename, index_col=0)
        df.index = df.index.astype(str)  # Convert index to string
    elif filename.endswith('.pkl'):
        df = pd.read_pickle(filename)
    # Perform a one-time conversion of datatype if needed
    if datatype is not None:
        try:
            datatype = [ast.literal_eval(col) for col in datatype]
        except (ValueError, SyntaxError):
            pass  # Keep datatype as is if conversion fails


    if outtype == 'list':
        if datatype is not None:
            if truedataindex is not None:
                var = [{col: df.loc[ti][col] for col in datatype} for ti in truedataindex]
            else:
                var = [{col: row[col] for col in datatype} for _, row in df.iterrows()]
        else:
            if truedataindex is not None:
                var = [{col: df.loc[ti][col] for col in df.columns} for ti in truedataindex]
            else:
                var = [{col: row[col] for col in df.columns} for _, row in df.iterrows()]

        return var

def read_data_csv(filename, datatype, truedataindex):
    """
    Read observational data from CSV files (legacy function).
    
    This is a legacy function for reading CSV files with flexible header configurations.
    Supports files with column headers, row headers, both, or neither. Handles missing
    values by replacing them with 'n/a'.
    
    Parameters
    ----------
    filename : str
        Path to the CSV file.
    datatype : list of str
        Column names (or positional column identifiers) for data types to extract.
    truedataindex : list
        Row identifiers where observational data was recorded (e.g., time stamps,
        observation indices). Used to select specific rows from the CSV.
    
    Returns
    -------
    imported_data : list of list
        2D list where each sublist represents a row of extracted data.
        Each element is either a float (numeric data) or string (text/missing data).
        Missing numeric values are replaced with 'n/a'.
    
    Notes
    -----
    - If the first column is 'header_both', the CSV is assumed to have both
      row and column headers.
    - If row count matches len(truedataindex), assumes column headers exist.
    - If row count is len(truedataindex)+1, assumes first row was misinterpreted
      as header and re-reads it as data.
    - NaN values in numeric columns are replaced with 'n/a' strings.
    
    See Also
    --------
    read_data_df : Modern version using pandas DataFrames with more flexible output.
    """

    df = pd.read_csv(filename)  # Read the file

    imported_data = []  # Initialize the 2D list of csv data
    tlength = len(truedataindex)
    dnumber = len(datatype)

    if df.columns[0] == 'header_both':  # csv file has column and row headers
        pos = [None] * dnumber
        for col in range(dnumber):
            # find index of data type in csv file header
            pos[col] = df.columns.get_loc(datatype[col])
        for t in truedataindex:
            row = df[df['header_both'] == t]  # pick row corresponding to truedataindex
            row = row.values[0]  # select the values of the dataframe row
            csv_data = [None] * dnumber
            for col in range(dnumber):
                if (not type(row[pos[col]]) == str) and (np.isnan(row[pos[col]])):  # do not check strings
                    csv_data[col] = 'n/a'
                else:
                    try:  # Making a float
                        csv_data[col] = float(row[pos[col]])
                    except:  # It is a string
                        csv_data[col] = row[pos[col]]
            imported_data.append(csv_data)
    else:  # No row headers (the rows in the csv file must correspond to the order in truedataindex)
        if tlength == df.shape[0]:  # File has column headers
            pos = [None] * dnumber
            for col in range(dnumber):
                # Find index of the header in datatype
                pos[col] = df.columns.get_loc(datatype[col])
        # File has no column headers (columns must correspond to the order in datatype)
        elif tlength == df.shape[0]+1:
            # First row has been misinterpreted as header, so we read first row again:
            temp = pd.read_csv(filename, header=None, nrows=1).values[0]
            pos = list(range(df.shape[1]))  # Assume the data is in the correct order
            csv_data = [None] * len(temp)
            for col in range(len(temp)):
                if (not type(temp[col]) == str) and (np.isnan(temp[col])):  # do not check strings
                    csv_data[col] = 'n/a'
                else:
                    try:  # Making a float
                        csv_data[col] = float(temp[col])
                    except:  # It is a string
                        csv_data[col] = temp[col]
            imported_data.append(csv_data)

        for rows in df.values:
            csv_data = [None] * dnumber
            for col in range(dnumber):
                if (not type(rows[pos[col]]) == str) and (np.isnan(rows[pos[col]])):  # do not check strings
                    csv_data[col] = 'n/a'
                else:
                    try:  # Making a float
                        csv_data[col] = float(rows[pos[col]])
                    except:  # It is a string
                        csv_data[col] = rows[pos[col]]
            imported_data.append(csv_data)

    return imported_data


def read_var_csv(filename, datatype, truedataindex):
    """
    Read variance/uncertainty data from CSV files (legacy function).
    
    This is a legacy function for reading CSV files containing variance or
    standard deviation data. Assumes that variance data is stored in alternating
    columns: data type identifier (string) followed by variance value (numeric).
    
    Parameters
    ----------
    filename : str
        Path to the CSV file containing variance data.
    datatype : list of str
        Column names (or positional identifiers) for data types. The function
        expects variance values in adjacent columns (datatype_col + 1).
    truedataindex : list
        Row identifiers where variance data was recorded. Used to select
        specific rows from the CSV.
    
    Returns
    -------
    imported_var : list of list
        2D list where each sublist contains alternating data type identifiers
        (strings, converted to lowercase) and variance values (floats).
        Format: [type1, var1, type2, var2, ...] for each row.
    
    Notes
    -----
    - The function expects variance data in alternating columns with the structure:
      [type_name, variance_value, type_name, variance_value, ...]
    - Data type names are automatically converted to lowercase.
    - Supports the same header configurations as read_data_csv:
      both headers, column headers only, row headers only, or no headers.
    - If first column is 'header_both', assumes both row and column headers exist.
    
    See Also
    --------
    read_var_df : Modern version using pandas DataFrames.
    read_data_csv : Companion function for reading observational data.
    """

    df = pd.read_csv(filename)  # Read the file

    imported_var = []  # Initialize the 2D list of csv data
    tlength = len(truedataindex)
    dnumber = len(datatype)

    if df.columns[0] == 'header_both':  # csv file has column and row headers
        pos = [None] * dnumber
        for col in range(dnumber):
            # find index of data type in csv file header
            pos[col] = df.columns.get_loc(datatype[col])
        for t in truedataindex:
            row = df[df['header_both'] == t]  # pick row
            row = row.values[0]  # select the values of the dataframe
            csv_data = [None] * 2 * dnumber
            for col in range(dnumber):
                csv_data[2*col] = row[pos[col]]
                try:  # Making a float
                    csv_data[2*col+1] = float(row[pos[col]]+1)
                except:  # It is a string
                    csv_data[2*col+1] = row[pos[col]+1]
            # Make sure the string input is lowercase
            csv_data[0::2] = [x.lower() for x in csv_data[0::2]]
            imported_var.append(csv_data)
    else:  # No row headers (the rows in the csv file must correspond to the order in truedataindex)
        if tlength == df.shape[0]:  # File has column headers
            pos = [None] * dnumber
            for col in range(dnumber):
                # Find index of datatype in csv file header
                pos[col] = df.columns.get_loc(datatype[col])
        # File has no column headers (columns must correspond to the order in datatype)
        elif tlength == df.shape[0]+1:
            # First row has been misinterpreted as header, so we read first row again:
            temp = pd.read_csv(filename, header=None, nrows=1).values[0]
            # Make sure the string input is lowercase
            temp[0::2] = [x.lower() for x in temp[0::2]]
            # Assume the data is in the correct order
            pos = list(range(0, df.shape[1], 2))
            csv_data = [None] * len(temp)
            for col in range(dnumber):
                csv_data[2 * col] = temp[2 * col]
                try:  # Making a float
                    csv_data[2*col+1] = float(temp[2*col+1])
                except:  # It is a string
                    csv_data[2*col+1] = temp[2*col+1]
            imported_var.append(csv_data)

        for rows in df.values:
            csv_data = [None] * 2 * dnumber
            for col in range(dnumber):
                csv_data[2*col] = rows[2*col]
                try:  # Making a float
                    csv_data[2*col+1] = float(rows[pos[col]+1])
                except:  # It is a string
                    csv_data[2*col+1] = rows[pos[col]+1]
            # Make sure the string input is lowercase
            csv_data[0::2] = [x.lower() for x in csv_data[0::2]]
            imported_var.append(csv_data)

    return imported_var


import os
from copy import deepcopy

from pipt.misc_tools.wavelet_tools import SparseRepresentation
from misc.structures import PETDataFrame

class DataReader:

    def __init__(self, info: dict, **kwargs):
        self.info = info
        self.data = info.get('truedata', info.get('data', None))
        self.var = info.get('datavar', info.get('var', None))

        # NB: Not sure if this will be used or needed!
        self.assimindex  = info.get('assimindex', None)
        self.truedataindex = info.get('truedataindex', None)
        self.datatype = info.get('datatype', None)

        # Sparse infor for data compression (for seismic data)
        self.sparse = info.get('sparse_info', None)
        self.sparse_data = []

        # Error handling for missing data or variance
        if self.data is None:
            msg = "Data missing: 'truedata' or 'data' key is missing in info dictionary."
            raise ValueError(msg)
        if self.var is None:
            msg = "Variance missing: 'datavar' or 'var' key is missing in info dictionary."
            raise ValueError(msg)


    def get_data(self) -> PETDataFrame:
        if isinstance(self.data, str):
            df = self._read_from_file(self.data)
        elif isinstance(self.data, dict):
            df = self._read_from_dict(self.data)
        else:
            msg = f"Unsupported data type: {type(self.data)}. Expected str or dict."
            raise TypeError(msg)
        

        # Process each cell for potential npz files and apply wavelet compression if specified
        for i, idx in enumerate(df.index):
            for col in df.columns:
                cell = df.loc[idx, col]

                if isinstance(cell, str) and cell.endswith('.npz'):
                    npzfile = np.load(cell, allow_pickle=True)
                    cell = npzfile[npzfile.files[0]]

                if (self.sparse is not None) and (col in self.sparse['compress_data']):
                    cell = self._wavelet_compression(cell, vintage=i)
                
                # Store new value
                df.loc[idx, col] = cell

                    
        # NB: Not sure if this will be used or needed!
        self.datatype = df.columns.tolist()
        self.assimindex = np.arange(len(df.index)).tolist()
        self.truedataindex = df.index.tolist()        
        return df
    

    def get_variance(self, data_df: PETDataFrame, sparse_data: list=None) -> PETDataFrame:
        if isinstance(self.var, str):
            _df = self._read_from_file(self.var)
        else:
            msg = f"Unsupported variance type: {type(self.var)}. Expected str (file path)."
            raise TypeError(msg)

        # Fill in dataframe
        df = PETDataFrame(columns=data_df.columns, index=data_df.index)
        for i, idx in enumerate(data_df.index):
            for c, col in enumerate(data_df.columns):
                if data_df.loc[idx, col] is not None:

                    # Sparse stuff (for seismic data)
                    if (
                        (sparse_data is not None)
                        and (self.sparse is not None)
                        and (col in self.sparse['compress_data'])
                    ):
                        var = np.power(sparse_data[i].est_noise, 2)

                    else:
                        var = self._extract_cell_variance(
                            _df.loc[idx, col], 
                            data_df.loc[idx, col], 
                            i, 
                            c,
                        )

                    df.loc[idx, col] = var
                else:
                    df.loc[idx, col] = None
        
        # Mark as ensemble if specified in info
        if 'emp_cov' in self.info:
            if (self.info['emp_cov'] == 'yes') or (self.info['emp_cov'] == True):
                df.is_ensemble = True

        return df


    def _read_from_file(self, filepath: str) -> PETDataFrame:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.pkl':
            df = PETDataFrame.from_pickle(filepath)
        elif ext == '.csv':
            df = PETDataFrame.from_csv(filepath, index_col=0)
            df = df.astype(float, errors='ignore')
        elif ext == '.npz':
            data = dict(np.load(filepath, allow_pickle=True))
            df = self._read_from_dict(data)
        else:
            msg = f"Unsupported file type: {filepath}. Expected .csv, .pkl, or .npz."
            raise ValueError(msg)
        return df
    

    def _read_from_dict(self, data_dict: dict) -> PETDataFrame:
        index = data_dict.pop('index', None)
        index_name = data_dict.pop('index_name', self.info.get('obsname', None))
        df = PETDataFrame(data=data_dict, index=index)
        df.index.name = index_name
        return df
    
    
    def _extract_cell_variance(self, var_cell, data_cell, i, c):
        
        # Variance given as relative percentage (e.g., ['rel', 5] means 5% of the data value)
        if var_cell[0].lower() == 'rel':
            return (0.01*var_cell[1] * data_cell)**2
        
        # Variance given as absolute value (e.g., ['abs', 0.5] means a variance of 0.5). 
        # If the value is iterable, it is indexed by column.
        elif var_cell[0].lower() == 'abs':
            val = var_cell[1]
            if hasattr(val, '__iter__') and not isinstance(val, str):
                return val[c]
            else:
                return val

        # Variance given as empirical ensemble (e.g., ['emp', [300, 350, 244, ...]]).
        elif (var_cell[0].lower() == 'emp'):
            return var_cell[1]
        
        # Variance given as full covariance matrix (e.g., ['cd', 'covfile.npz']).
        elif (var_cell[0].lower() == 'cd') and (var_cell[1].endswith('.npz')):
            # Populate once
            if not hasattr(self, 'cov'):
                covfile = np.load(var_cell[1], allow_pickle=True)['cov']
                self.cov = covfile[covfile.files[0]]
            return self.cov[i*c, i*c]

        # Return None if no data for this cell
        elif data_cell is None:
            return None

        else:
            msg = f"Unsupported variance type in cell: {var_cell}. Expected format like ['rel', value], ['abs', values], or ['emp', value]."
            raise ValueError(msg)
    

    def _wavelet_compression(self, arr, vintage):
        
        options = deepcopy(self.sparse)
        options['mask'] = options['mask'][vintage]
        min_noise = options['min_noise']

        if isinstance(min_noise, list):
            if 0 <= vintage < len(min_noise):
                options['min_noise'] = min_noise[vintage]
            else:
                msg = 'min_noise must either be scalar or list with one number for each vintage'
                raise ValueError(msg)
        
        # Apply wavelet compression
        sparsrep = SparseRepresentation(options)
        arr_compressed, wdec_rec = sparsrep.compress(arr, th_mult=options['th_mult'])
        self.sparse_data.append(sparsrep) # Store the information

        # Save reconstructed data
        arr_reconstructed = sparsrep.reconstruct(wdec_rec)  # reconstruct the data
        np.savez('truedata_rec_' + str(vintage) + '.npz', arr_reconstructed)

        if self.sparse.get('use_ensemble', False):
            return arr
        else:
            return arr_compressed

