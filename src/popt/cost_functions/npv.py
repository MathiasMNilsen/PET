"""Net present value."""
import numpy  as np
import pandas as pd

DEFAULT_ECON = {
    'wop': 471.0,  # Oil price: $/Sm3 (equvalent to 75 $/STB)
    'wgp': 0.4,    # Gas price: $/Sm3
    'wwp': 40.0,   # Cost of water production per unit volume
    'wwi': 25.0,   # Cost of water injection per unit volume
    'disc': 0.08,  # Discount rate per year
}


def npv(pred_data: pd.DataFrame, **kwargs):
    # --- Extract economic parameters and scaling factor ---
    input_dict = kwargs.get("input_dict", {})
    econ_params = dict(input_dict.get("npv_const", DEFAULT_ECON))
    scaling_factor = econ_params.pop("obj_scaling", 1.0)

    # --- Compute incremental volumes ---
    vol_oil = pred_data["FOPT"].diff()
    vol_gas = pred_data["FGPT"].diff()
    vol_water_prod = pred_data["FWPT"].diff()
    vol_water_inj = pred_data["FWIT"].diff()

    # --- Compute time in years from start ---
    time_index = pred_data.index.to_numpy()
    years = (time_index - time_index[0]) / np.timedelta64(365, "D")

    # --- Compute revenue, costs and discounted cash flow ---
    revenue = vol_oil * econ_params["wop"] + vol_gas * econ_params["wgp"]
    operating_cost = (
        vol_water_prod * econ_params["wwp"]
        + vol_water_inj * econ_params["wwi"]
    )
    discount_factor = (1.0 + econ_params["disc"]) ** years
    discounted_cash_flow = (revenue - operating_cost) / discount_factor

    # --- Return scaled NPV ---
    return discounted_cash_flow.sum() / scaling_factor
