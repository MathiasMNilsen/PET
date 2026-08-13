"""Comprehensive tests for localization methods and facade behavior."""

import numpy as np
from  pipt.misc_tools.analysis_tools import truncSVD
from pipt.update_schemes.update_methods_ns import approx_update
from pipt.localization import (
    AutoAdaptiveLocalization,
    build_localization_instance,
)

np.random.seed(128928)  # For reproducibility

NX = 8
NY = 4
NE = 10

X = np.array([
    [1, 3, 2, 5, 4, 6, 7, 8, 9, 10],
    [2, 1, 4, 3, 6, 5, 8, 7, 10, 9],
    [5, 4, 6, 3, 7, 2, 8, 1, 10, 9],
    [3, 6, 2, 7, 1, 8, 4, 9, 5, 10],
    [7, 3, 8, 2, 9, 1, 10, 4, 6, 5],
    [1, 4, 3, 6, 2, 7, 5, 9, 8, 10],
    [8, 5, 9, 4, 10, 3, 7, 2, 6, 1],
    [4, 2, 6, 1, 7, 3, 8, 5, 10, 9],
], dtype=float) # shape: (NX, NE)

Y = np.array([
    [1, 2, 3, 5, 4, 6, 8, 7, 9, 10],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    [4, 6, 1, 8, 3, 7, 2, 10, 5, 9],
    [2, 8, 4, 7, 1, 9, 3, 6, 10, 5],
], dtype=float) # shape: (NY, NE)

X = X[:, :NE]  # shape: (NX, NE)
Y = Y[:, :NE]  # shape: (NY, NE)

# Correlation matrix
R = np.corrcoef(X, Y)[:NX, NX:] # Shape: (NX, NY)

def test_config_autoadaloc():
    loc_info = {
        "name": "autoadaloc",
        "field": [1, 5, 5],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 0.4,
        "type": "soft",
        "projection": "rank-r"
    }
    loc = build_localization_instance(loc_info)

    assert isinstance(loc, AutoAdaptiveLocalization)
    assert loc.name == "autoadaloc"
    assert loc.field == [1, 5, 5]
    assert loc.actnum is None
    assert loc.cutoff == 0.4
    assert loc.tapertype == "soft"
    assert loc.threshold == "fixed"


def test_autoadaloc_no_trunc():
    loc_info = {
        "name": "autoadaloc",
        "field": [4, 2],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 0.005,
        "type": "hard",
        "projection": "rank-r",
    }
    loc = AutoAdaptiveLocalization(loc_info)
    taper = loc(X, Y)
    assert taper.shape == (NX, NY)
    np.testing.assert_allclose(taper, np.ones((NX, NY)))


def test_autoadaloc_partial_trunc():
    loc_info = {
        "name": "autoadaloc",
        "field": [4, 2],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 0.4,
        "type": "hard",
        "projection": "rank-r"
    }
    loc = AutoAdaptiveLocalization(loc_info)
    taper_result = loc(X, Y)

    # Expected taper matrix
    taper_expected = np.where(np.abs(R) >= loc.cutoff, 1, 0)

    np.testing.assert_allclose(taper_result, taper_expected)


def test_autoadaloc_full_trunc():
    loc_info = {
        "name": "autoadaloc",
        "field": [4, 2],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 1.0,
        "type": "hard",
        "projection": "rank-r"
    }
    loc = AutoAdaptiveLocalization(loc_info)
    taper = loc(X, Y)
    assert taper.shape == (NX, NY)
    np.testing.assert_allclose(taper, np.zeros((NX, NY)))


def test_approx_update_with_autoadaloc():

    loc_info = {
        "name": "autoadaloc",
        "field": [4, 2],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 0.4,
        "type": "hard",
        "projection": "rank-r"
    }

    # Define ensemble matrices
    enX = X.copy()
    enY = Y.copy()
    enE = enY.mean(axis=1)[:, None] + np.random.normal(0, 0.1, size=enY.shape)
    Cdd = 0.1*np.ones(NY)

    # Define class
    class DummyApproxUpdate(approx_update):
        localization = AutoAdaptiveLocalization(loc_info)
        lam = 1.0
        trunc_energy = 0.98
        cov_data = Cdd
        keys_da = {"emp_cov": False}

    # Step with localization
    approx = DummyApproxUpdate()
    step_loc = approx.update(enX, enY, enE)

    # Step without localization
    approx_no_loc = DummyApproxUpdate()
    approx_no_loc.localization = type('localization', (object,), {'name': None})()
    step_no_loc = approx_no_loc.update(enX, enY, enE)

    # Calculate step manually without localization
    scy = np.sqrt(Cdd)
    PI = (np.eye(NE) - np.ones((NE, NE)) / NE)/ np.sqrt(NE-1)
    X_anom = enX @ PI
    Y_anom = (enY @ PI)  / scy[:, None]
    D_anom = (enE - enY) / scy[:, None]
    Ur, Sr, VrT = truncSVD(Y_anom, energy=0.98)
    X1 = Ur.T @ D_anom
    X2 = X1 / (1 + 1.0 + Sr**2)[:, None]
    X3 = VrT.T @ np.diag(Sr) @ X2 
    step_expected_no_loc = X_anom @ X3

    # Calculate step manually with localization
    loc = AutoAdaptiveLocalization(loc_info)
    Y_anom_proj = np.diag(Sr) @ VrT
    taper = loc(X=X_anom, Y=Y_anom_proj)
    Cxy_loc = taper * (X_anom @ Y_anom_proj.T)
    step_loc_expected = Cxy_loc @ X2

    np.testing.assert_allclose(step_loc, step_loc_expected)
    np.testing.assert_allclose(step_no_loc, step_expected_no_loc)
    assert not np.array_equal(step_loc, step_no_loc)


def compares_with_old_autoadaloc():
    
    loc_info = {
        "name": "autoadaloc",
        "field": [4, 2],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 0.7,
        "type": "hard",
        "projection": "rank-r"
    }

    # Define ensemble matrices
    enX = X.copy()
    enY = Y.copy()
    enE = enY.mean(axis=1)[:, None] + np.random.normal(0, 0.1, size=enY.shape)
    Cdd = 0.1*np.ones(NY)

    # --------------------------------------------------------
    # Step with localization (using approx_update)
    # --------------------------------------------------------
    scy = np.sqrt(Cdd)
    PI = (np.eye(NE) - np.ones((NE, NE)) / NE)/ np.sqrt(NE-1)
    X_anom = enX @ PI
    Y_anom = (enY @ PI)  / scy[:, None]
    D_anom = (enE - enY) / scy[:, None]
    Ur, Sr, VrT = truncSVD(Y_anom, energy=0.98)
    X1 = Ur.T @ D_anom
    X2 = X1 / (1 + 1.0 + Sr**2)[:, None]
    loc = AutoAdaptiveLocalization(loc_info)
    Y_anom_proj = np.diag(Sr) @ VrT
    Cxy_loc = loc(X=X_anom, Y=Y_anom_proj)
    step_loc = Cxy_loc @ X2

    # --------------------------------------------------------
    # Step with no localization (using approx_update)
    # --------------------------------------------------------
    scy = np.sqrt(Cdd)
    PI = (np.eye(NE) - np.ones((NE, NE)) / NE)/ np.sqrt(NE-1)
    X_anom = enX @ PI
    Y_anom = (enY @ PI)  / scy[:, None]
    D_anom = (enE - enY) / scy[:, None]
    Ur, Sr, VrT = truncSVD(Y_anom, energy=0.98)
    X1 = Ur.T @ D_anom
    X2 = X1 / (1 + 1.0 + Sr**2)[:, None]
    X3 = VrT.T @ np.diag(Sr) @ X2 
    step_no_loc = X_anom @ X3


    # --------------------------------------------------------
    # Old step with localization
    # --------------------------------------------------------
    loc = AutoAdaptiveLocalization(loc_info)
    scy = np.sqrt(Cdd)
    PI = (np.eye(NE) - np.ones((NE, NE)) / NE)/ np.sqrt(NE-1)
    X_anom = enX @ PI
    Y_anom = (enY @ PI)  / scy[:, None]
    D_anom = (enE - enY) / scy[:, None]
    Ur, Sr, VrT = truncSVD(Y_anom, energy=0.98)
    reg_term = np.eye(Sr.size) + np.diag(Sr**2)
    X2 = VrT.T @ np.diag(Sr) @ np.linalg.solve(reg_term, Ur.T)

    corr = loc.corr_matrix(X_anom, X2 @ D_anom)
    T = np.where(np.abs(corr) >= loc.cutoff, 1, 0)
    step_old_loc = (T * X_anom) @ (X2 @ D_anom)

    loc.projection = 'ensemble'
    step_old_loc_2 = loc(
        X=X_anom,               # shape: (nx, ne)
        Y=X2 @ D_anom           # shape: (ne, ne)
    )
    print(step_old_loc_2-step_old_loc)
    # --------------------------------------------------------


    # --------------------------------------------------------
    # Comupare the full loc update
    # --------------------------------------------------------
    X_anom = enX @ PI
    Y_anom = (enY @ PI) 
    D_anom = (enE - enY)

    # Kalman gain with localization
    loc = AutoAdaptiveLocalization(loc_info)
    corr = np.corrcoef(X_anom, Y_anom)[:NX, NX:]
    T = np.where(np.abs(corr) >= loc.cutoff, 1, 0)
    Cxy = T * (X_anom @ Y_anom.T)
    CYY = Y_anom @ Y_anom.T
    step_loc_full = Cxy @ np.linalg.solve(CYY + np.diag(Cdd), D_anom)

    # --------------------------------------------------------
    # full step without localization
    step_no_loc_full = (X_anom @ Y_anom.T) @ np.linalg.solve(Y_anom @ Y_anom.T + np.diag(Cdd), D_anom)
    

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # --------------------------------------------------------
    # Common color scale (symmetric around zero)
    # --------------------------------------------------------
    vabs = np.max([
        np.abs(step_no_loc).max(),
        np.abs(step_loc).max(),
        np.abs(step_old_loc).max(),
        np.abs(step_loc_full).max(),
        np.abs(step_no_loc_full).max(),
    ])

    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))

    # Flatten for easier indexing
    ax = ax.ravel()

    im0 = ax[0].imshow(
        step_no_loc,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
    )

    im1 = ax[1].imshow(
        step_loc,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
    )

    im2 = ax[2].imshow(
        step_old_loc,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
    )

    im3 = ax[3].imshow(
        step_no_loc_full,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
    )

    im4 = ax[4].imshow(
        step_loc_full,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
    )

    # Optional: show difference between new and old localization
    im5 = ax[5].imshow(
        step_loc - step_old_loc,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
    )

    titles = [
        "Approx. Update (No Loc)",
        "Approx. Update (New Loc)",
        "Approx. Update (Old Loc)",
        "Full Update (No Loc)",
        "Full Update (Loc)",
        "New Loc − Old Loc",
    ]

    for a, title in zip(ax, titles):
        a.set_title(title)
        a.set_xlabel("ensemble members")
        a.set_ylabel("state variables")

    # Colorbar
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])

    cbar = fig.colorbar(im0, cax=cax)
    cbar.set_label("Update value")

    plt.show()


    corr_new = np.corrcoef(
        step_loc.ravel(),
        step_loc_full.ravel()
    )[0, 1]

    corr_old = np.corrcoef(
        step_old_loc.ravel(),
        step_loc_full.ravel()
    )[0, 1]

    print(f"New loc correlation: {corr_new:.4f}")
    print(f"Old loc correlation: {corr_old:.4f}")




#compares_with_old_autoadaloc()



