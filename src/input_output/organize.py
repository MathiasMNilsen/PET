"""Descriptive description."""

from copy import deepcopy
from pathlib import Path
import csv
import datetime as dt
import os
import pandas as pd
import yaml



class ConfigNormalizer:
    """
    Utility class for normalizing and type-converting configuration dictionaries for PIPT/POPT workflows.

    This class provides static methods to process and normalize configuration sections such as 'datatype',
    'truedataindex', 'reportpoint', and 'assimindex'.
    """

    @staticmethod
    def normalize_datatype(datatype):
        """
        Normalize the 'datatype' field: read from CSV if needed, ensure list of strings.
        """
        if isinstance(datatype, str) and datatype.endswith('.csv'):
            with open(datatype) as csvfile:
                reader = csv.reader(csvfile)
                return [str(col) for row in reader for col in row]
        if not isinstance(datatype, list):
            return [datatype]
        return [str(x) for x in datatype]

    @staticmethod
    def normalize_truedataindex(truedataindex):
        """
        Normalize the 'truedataindex' field: read from CSV if needed, ensure list of ints.
        """
        if isinstance(truedataindex, str) and truedataindex.endswith('.csv'):
            with open(truedataindex) as csvfile:
                reader = csv.reader(csvfile)
                return [int(col) for row in reader for col in row]
        if not isinstance(truedataindex, list):
            return [truedataindex]
        return [int(x) for x in truedataindex]

    @staticmethod
    def normalize_reportpoint(reportpoint):
        """
        Normalize the 'reportpoint' field: handle CSV, dict (date_range), or pass through.
        """
        if isinstance(reportpoint, str):
            return report_point_file_reader(reportpoint)
        elif isinstance(reportpoint, dict):
            return pd.date_range(**reportpoint).to_pydatetime().tolist()
        elif not isinstance(reportpoint, list):
            return [reportpoint]
        return reportpoint

    @staticmethod
    def normalize_assimindex(assimindex):
        """
        Normalize the 'assimindex' field: read from CSV if needed, ensure list of lists of ints.
        """
        if isinstance(assimindex, str) and assimindex.endswith('.csv'):
            with open(assimindex) as csvfile:
                reader = csv.reader(csvfile)
                return [[int(col) for col in row] for row in reader]
        if not isinstance(assimindex, list):
            return [assimindex]
        # If it's a flat list, wrap in another list
        if assimindex and not isinstance(assimindex[0], list):
            return [assimindex]
        return assimindex

    @staticmethod
    def normalize_config(keys_pr, keys_fwd, keys_en=None):
        """
        Normalize all relevant fields in the config dictionaries and return new dicts.
        """
        keys_pr = deepcopy(keys_pr) if keys_pr else {}
        keys_fwd = deepcopy(keys_fwd) if keys_fwd else {}
        keys_en = deepcopy(keys_en) if keys_en else {} if keys_en is not None else None

        # Normalize datatype
        if 'datatype' in keys_fwd:
            keys_fwd['datatype'] = ConfigNormalizer.normalize_datatype(keys_fwd['datatype'])
            keys_pr['datatype'] = keys_fwd['datatype']

        # Normalize truedataindex
        if 'truedataindex' in keys_pr:
            keys_pr['truedataindex'] = ConfigNormalizer.normalize_truedataindex(keys_pr['truedataindex'])

        # Normalize reportpoint
        if 'reportpoint' in keys_fwd:
            keys_fwd['reportpoint'] = ConfigNormalizer.normalize_reportpoint(keys_fwd['reportpoint'])

        # Normalize assimindex
        if 'assimindex' in keys_pr:
            keys_pr['assimindex'] = ConfigNormalizer.normalize_assimindex(keys_pr['assimindex'])

        return keys_pr, keys_fwd, keys_en


def report_point_file_reader(filepath):
    """
    Reads a file containing report points and returns a list of parsed values as integers or datetimes.

    Supported file types:
        - CSV (.csv): Each cell is parsed as an integer if possible, otherwise as a datetime (supports ISO and common formats).
        - TXT (.txt): Each line is parsed as an ISO 8601 datetime string.
        - YAML (.yaml): Each entry is parsed as a datetime (supports ISO and common formats).

    Parameters
    ----------
    filepath : str
        Path to the input file. Must exist and have a supported extension (.csv, .txt, .yaml).

    Returns
    -------
    list
        List of parsed report points. Elements are either int or pandas.Timestamp/datetime.datetime objects,
        depending on the file content.

    Raises
    ------
    AssertionError
        If the file does not exist.
    ValueError
        If the file extension is not supported.

    Notes
    -----
    - Empty cells in CSV files are skipped.
    - For CSV and YAML, pandas.to_datetime is used for flexible datetime parsing.
    - For TXT, each line must be a valid ISO 8601 datetime string.
    """
    assert os.path.isfile(filepath), f"File {filepath} does not exist."
    if Path(filepath).suffix.lower() == ".csv":
        df = pd.read_csv(filepath, header=None)
        values = df.values.ravel()
        rpoints = []
        for v in values:
            if pd.isna(v):
                continue  # skip empty cells
            try:
                rpoints.append(int(v))
            except (ValueError, TypeError):
                rpoints.append(pd.to_datetime(v))

    elif Path(filepath).suffix.lower() == ".txt":
        with open(filepath) as file:
            rpoints = [dt.datetime.fromisoformat(line.strip()) for line in file]
    elif Path(filepath).suffix.lower() == ".yaml":
        with open(filepath) as file:
            rpoints = [pd.to_datetime(v) for v in yaml.safe_load(file)]
    else:
        raise ValueError(f"Unsupported file type: {filepath}")
    return rpoints