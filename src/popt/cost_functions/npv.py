"""Net present value."""
import numpy  as np
import pandas as pd

_DEFAULT_ECON = {
    'wop': 471.0,  # Oil price: $/Sm3 (equvalent to 75 $/STB)
    'wgp': 0.4,    # Gas price: $/Sm3
    'wwp': 40.0,   # Cost of water production per unit volume
    'wwi': 25.0,   # Cost of water injection per unit volume
    'disc': 0.08,  # Discount rate per year
}


def npv(pred_data: pd.DataFrame, **kwargs):
    # Economic values
    kw = kwargs.get('input_dict', {})
    econ = dict(kw.get('npv_const', _DEFAULT_ECON))
    scaling = econ.pop('obj_scaling', 1.0)

    # Calculate the change in volumes for each time step
    volOil = pred_data['FOPT'].diff()
    volGas = pred_data['FGPT'].diff()
    volWPR = pred_data['FWPT'].diff()
    volWIN = pred_data['FWIT'].diff()

    # Get dates and calculate days (Assuming index is datetime)
    dates = pred_data.index.values
    years = (dates - dates[0]) / np.timedelta64(365, 'D')

    # Calculate the NPV
    revenue = volOil * econ['wop'] + volGas * econ['wgp']
    cost = volWPR * econ['wwp'] + volWIN * econ['wwi']
    npvval = (revenue - cost) / ((1 + econ['disc']) ** years)

    return npvval.sum() / scaling