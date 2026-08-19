"""Generate docs/tutorials/popt/tutorial_popt.ipynb.

Written as a generator rather than by hand-editing JSON so the cell sources stay
readable and reviewable in one place.
"""

import json
from pathlib import Path

OUT = Path("docs/tutorials/popt/tutorial_popt.ipynb")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = []

# ----------------------------------------------------------------------
cells.append(md("""\
# Tutorial for running the Python Optimization Toolbox (POPT)

<font size=4em>As an illustrative example we choose a 2D-field with one producer and two (water) injectors. The figure below shows the permeability field and the well positions. The grid is 100x100, and the porosity is 0.18. The optimization problem is to find the bottom-hole pressure control for each well that maximizes the net present value (NPV) over the production period.

<img src="../permx.png" alt="drawing" width="500"/>
<br>
<font size=4em>POPT mirrors PIPT: an *ensemble* object owns the control perturbations and the gradient estimate, and an *optimizer* owns its own iteration loop. The first step is to load the necessary external and local modules.
"""))

cells.append(code("""\
# Import global modules
import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import local modules
from input_output import read_config                 # the config reader
from popt.ensembles import GaussianEnsemble          # control perturbations and gradients
from popt.optimization_methods import EnOpt, SmcOpt  # the optimizers; each owns its loop
from subsurface.multphaseflow.opm import flow        # the simulator we want to use
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Set the random seed:
"""))

cells.append(code("""\
np.random.seed(101122)
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Each simulator call runs in its own <span style="font-family:Courier;">En_&lt;member&gt;</span> folder, which it creates with <span style="font-family:Courier;">os.mkdir</span> &mdash; so a folder left behind by an interrupted run makes the next one fail with <span style="font-family:Courier;">FileExistsError</span>. PET clears them when an ensemble is constructed, but not between runs, so we define a helper and call it before each optimization. That keeps the run cells safe to re-execute on their own.
"""))

cells.append(code("""\
def clean_run_folders(*result_folders):
    \"\"\"Remove simulator scratch folders, and any results being replaced.\"\"\"
    for folder in glob('En_*'):
        shutil.rmtree(folder, ignore_errors=True)
    for folder in result_folders:
        shutil.rmtree(folder, ignore_errors=True)
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Read the input file. In this tutorial the input file is written as a .toml file, and consists of three main keys: <span style="font-family:Courier;">ensemble</span>, <span style="font-family:Courier;">optim</span> and <span style="font-family:Courier;">fwdsim</span>. The first contains keys related to the ensemble of control perturbations, the second the options for the optimization algorithm, and the third the options for the forward simulation model and the objective function.
"""))

cells.append(code("""\
!cat init_optim.toml
ko, kf, ke = read_config.read('init_optim.toml')
# ko  -->  Optimization settings
# kf  -->  Simulator settings
# ke  -->  Ensemble settings
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Set the initial controls. Note that the filenames correspond to the <span style="font-family:Courier;">mean</span> entries given in the input file above. The two injectors start at 300 and 250 bar, the producer at 100 bar.
"""))

cells.append(code("""\
np.savez('init_injbhp.npz', np.array([300.0, 250.0]))
np.savez('init_prodbhp.npz', np.array([100.0]))
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Define the objective function. This is the one piece POPT does not supply: you hand it any callable that takes the simulated data and returns a scalar to be **minimized**. Here it is the discounted net present value, with the economic constants read from the <span style="font-family:Courier;">npv_const</span> block of the input file.

<font size=4em>Note the <span style="font-family:Courier;">obj_scaling</span> of -1e6: the negative sign turns maximizing NPV into a minimization, and the 1e6 puts the value in millions so the optimizer works on a sensible scale.
"""))

cells.append(code("""\
DEFAULT_ECON = {
    'wop': 471.0,  # Oil price: $/Sm3 (equivalent to 75 $/STB)
    'wgp': 0.4,    # Gas price: $/Sm3
    'wwp': 40.0,   # Cost of water production per unit volume
    'wwi': 25.0,   # Cost of water injection per unit volume
    'disc': 0.08,  # Discount rate per year
}


def npv(pred_data: pd.DataFrame, **kwargs):
    \"\"\"Discounted net present value of one simulated production profile.\"\"\"
    # Economic parameters, from the config's npv_const block if present
    input_dict = kwargs.get('input_dict', {})
    econ = dict(input_dict.get('npv_const', DEFAULT_ECON))
    scaling_factor = econ.pop('obj_scaling', 1.0)

    # Incremental volumes per report step
    vol_oil = pred_data['FOPT'].diff()
    vol_gas = pred_data['FGPT'].diff()
    vol_water_prod = pred_data['FWPT'].diff()
    vol_water_inj = pred_data['FWIT'].diff()

    # Time in years since the start of the run
    time_index = pred_data.index.to_numpy()
    years = (time_index - time_index[0]) / np.timedelta64(365, 'D')

    # Revenue, cost, and discounting
    revenue = vol_oil * econ['wop'] + vol_gas * econ['wgp']
    operating_cost = vol_water_prod * econ['wwp'] + vol_water_inj * econ['wwi']
    discount_factor = (1.0 + econ['disc']) ** years

    return ((revenue - operating_cost) / discount_factor).sum() / scaling_factor
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Initialize the ensemble with the ensemble keys, the simulator and the objective function, then extract the initial control vector (<span style="font-family:Courier;">x0</span>), its covariance (<span style="font-family:Courier;">cov</span>) and the bounds. The ensemble is what turns a non-differentiable simulator into something gradient-based methods can use: it perturbs the controls, runs the simulator on each perturbation, and forms an ensemble approximation of the gradient.
"""))

cells.append(code("""\
sim = flow(kf)
ensemble = GaussianEnsemble(ke, sim, npv)

x0 = ensemble.get_state()
cov = ensemble.get_cov()
bounds = ensemble.get_bounds()

print(f'controls: {x0}')
print(f'bounds:   {bounds}')
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Run the optimization with EnOpt. During the run, useful information is written to the screen and to a log file. As in PIPT, there are two ways to do this &mdash; the class-level shortcut that constructs and runs in one call, or an instance you keep and drive yourself.

<font size=4em>Only the gradient is passed here, because the input file sets <span style="font-family:Courier;">hessian = false</span>. To use a second-order search direction, set that key to true and pass <span style="font-family:Courier;">hess=ensemble.hessian</span> as well &mdash; note that the Hessian is evaluated whenever it is supplied, so passing it while the key is false costs simulator runs for nothing.
"""))

cells.append(code("""\
clean_run_folders(ko.get('savefolder', 'Iteration_Results'))

# There are two ways to run the optimization:

# Option 1: the class-level shortcut, when the optimizer object is not needed afterwards
res_enopt = EnOpt.minimize(
    x0=x0,
    fun=ensemble.function,
    jac=ensemble.gradient,
    args=(cov,),
    bounds=bounds,
    **ko,
)

# Option 2: keep the optimizer, then run it
# enopt = EnOpt(fun=ensemble.function, x=x0, jac=ensemble.gradient,
#               args=(cov,), bounds=bounds, **ko)
# res_enopt = enopt.run_optimization()

print(f'NPV: {-res_enopt.fun:.1f} million $ after {res_enopt.nit} iterations')
print(res_enopt)
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Plot the objective function against iteration. The optimizer writes one file per iteration, <span style="font-family:Courier;">optimize_result_{i}.npz</span>, into the folder named by the <span style="font-family:Courier;">savefolder</span> key &mdash; the counterpart of PIPT's <span style="font-family:Courier;">assimilation_result_{i}.npz</span>. Saving happens only when <span style="font-family:Courier;">saveit</span> is true.
"""))

cells.append(code("""\
def read_npv_history(folder):
    \"\"\"Collect the NPV at each iteration from the saved result files.\"\"\"
    values = []
    it = 0
    while True:
        file = f'{folder}/optimize_result_{it}.npz'
        if not os.path.exists(file):
            break
        info = np.load(file)
        # 'fun' is the objective value at that iteration. The sign flip undoes
        # the negative obj_scaling, turning the minimized quantity back into NPV.
        values.append(-float(np.mean(info['fun'])))
        it += 1
    return values


npv_enopt = read_npv_history(ko.get('savefolder', 'Iteration_Results'))

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor='white')
ax.plot(npv_enopt, 's-', color='#4C78A8', linewidth=2, markersize=7, label='EnOpt')
ax.set_xlabel('Iteration no.', size=13)
ax.set_ylabel('NPV [million $]', size=13)
ax.set_title('Objective function', size=14)
ax.set_xticks(range(len(npv_enopt)))
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>The same problem with a different optimizer. <span style="font-family:Courier;">SmcOpt</span> is a sequential Monte Carlo method: it needs no gradient, taking a weighting function instead of a Jacobian. Everything else &mdash; the ensemble, the objective, the bounds &mdash; is reused unchanged, which is the point of keeping the optimizer separate from the ensemble.

<font size=4em>Both runs start from the same <span style="font-family:Courier;">x0</span> captured above, so the comparison is fair. Note that <span style="font-family:Courier;">ensemble.get_state()</span> would <em>not</em> do here: it returns the ensemble's current controls, which the first optimization has already moved.
"""))

cells.append(code("""\
from copy import deepcopy

ko_smc = deepcopy(ko)
ko_smc['savefolder'] = 'Results_smc'   # keep EnOpt's files for the comparison below

clean_run_folders(ko_smc['savefolder'])

res_smc = SmcOpt.minimize(
    x0=x0,
    fun=ensemble.function,
    sens=ensemble.calc_ensemble_weights,
    args=(cov,),
    bounds=bounds,
    **ko_smc,
)

print(f'NPV: {-res_smc.fun:.1f} million $ after {res_smc.nit} iterations')
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Compare the two:
"""))

cells.append(code("""\
npv_smc = read_npv_history(ko_smc['savefolder'])

fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor='white')
ax.plot(npv_enopt, 's-', color='#4C78A8', linewidth=2, markersize=7, label='EnOpt')
ax.plot(npv_smc, 'o--', color='#E45756', linewidth=2, markersize=7, label='SmcOpt')
ax.set_xlabel('Iteration no.', size=13)
ax.set_ylabel('NPV [million $]', size=13)
ax.set_title('EnOpt vs. SmcOpt', size=14)
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
## Setting up the .mako file
<font size=4em>The optimization relies on a .mako file for writing the current control variables to the flow simulator input. In this case, the flow simulator is opm-flow [opm-projects.org](opm-projects.org), and the input file is provided as a text file (.DATA file). The .mako file is created by replacing the keywords <span style="font-family:Courier;">WCONINJE</span> and <span style="font-family:Courier;">WCONPROD</span> in the .DATA file with:

    WCONINJE
    'INJ-1'  WATER 'OPEN' BHP 2* ${injbhp[0]} /
    'INJ-2'  WATER 'OPEN' BHP 2* ${injbhp[1]} /
    /

    WCONPROD
     'PRO-1' 'OPEN' BHP 5* ${prodbhp[0]} /
    /

<font size=4em>The names <span style="font-family:Courier;">injbhp</span> and <span style="font-family:Courier;">prodbhp</span> are the entries of the <span style="font-family:Courier;">state</span> key in the input file, so the .mako placeholders and the config have to agree.
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
## Running locally

<font size=4em>It is recommended to run the notebook from a virtual environment. Follow these steps to run this notebook on your own computer:

<font size=4em>*Step 1: Create virtual environment as normal*

    python3 -m venv pet_venv

<font size=4em>Then activate the environment using:

    source pet_venv/bin/activate

<font size=4em>*Step 2: Install Jupyter Notebook into virtual environment*

    python3 -m pip install ipykernel

<font size=4em>*Step 3: Install PET in the virtual environment, see [PET installation](https://github.com/Python-Ensemble-Toolbox/PET)*

<font size=4em>*Step 4: Allow Jupyter access to the kernel within the virtual environment*

    python3 -m ipykernel install --user --name=pet_venv

<font size=4em>Start jupyter notebook, and load tutorial_popt.ipynb (this file). On the jupyter notebook toolbar, select 'Kernel' and 'Change Kernel'.
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

OUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {OUT} with {len(cells)} cells")
