"""EnRML (IES) without the prior increment term."""

import numpy as np
from copy import deepcopy
import copy as cp
from scipy.linalg import solve, sqrtm
import pickle
import warnings

import pipt.misc_tools.ensemble_tools as entools
import pipt.misc_tools.analysis_tools as at
import pipt.misc_tools.extract_tools as extract

from pipt.localization import _calc_loc


class approx_update():
    """
    Approximate LM Update scheme as defined in "Chen, Y., & Oliver, D. S. (2013). Levenberg–Marquardt forms of the iterative ensemble
    smoother for efficient history matching and uncertainty quantification. Computational Geosciences, 17(4), 689–703.
    https://doi.org/10.1007/s10596-013-9351-5". Note that for a EnKF or ES update, or for update within GN scheme, lambda = 0.
    """

    def update(self, enX, enY, enE, **kwargs):
        ''' 
        Perform the approximate LM update.

        Parameters:
        ----------
            enX : np.ndarray 
                State ensemble matrix (nx, ne)
            
            enY : np.ndarray
                Predicted data ensemble matrix (nd, ne)
            
            enE : np.ndarray
                Ensemble of perturbed observations (nd, ne)
        '''
        # Shapes
        nx, ne = enX.shape
        ny, _  = enY.shape

        # Scaling factors and other attributes needed for the update
        cov = getattr(self, 'cov_data', np.eye(ny))  # Data covariance matrix (ny,ny) or (ny,)
        scx = getattr(self, 'scale_state', np.ones(nx))
        scy = getattr(self, 'scale_data', self.sqrtm(cov))
        PI  = getattr(
            self, 'proj', 
            (np.eye(ne) - np.ones((ne, ne)) / ne)/ np.sqrt(ne-1)
        )  # shape: (ne, ne) such that A@PI = A - mean(A)/sqrt(ne-1) for any ensemble matrix A of shape (na, ne) 

        # Check for adjoint-based update
        if kwargs.get('enAdj', None):
            Y = kwargs['enAdj'].mean(axis=-1) @ enX @ PI    # shape: (nd, ne)
        else:
            Y = enY @ PI                                    # shape: (nd, ne) --> Such that Cyy ≈ Y @ Y.T

        # Anomaly matrices
        X_anom = self.solve(scx, enX @ PI)                  # shape: (nx, ne) --> State anomalies: (X-mean(X))/sqrt(ne-1)
        Y_anom = self.solve(scy, Y)                         # shape: (nd, ne) --> Predicted data anomalies: (Y-mean(Y))/sqrt(ne-1)
        D_anom = self.solve(scy, enE - enY)                 # shape: (nd, ne) --> Innovation ensemble: data - predictions

        # Truncated SVD on predicted data anomalies
        Ur, Sr, VrT = at.truncSVD(Y_anom, energy=self.trunc_energy) # shape: (nd, nr), (nr,), (nr, ne)

        # ===============================================
        # Compute step
        # ===============================================
        X1 = Ur.T @ D_anom                                  # shape: (nr, ne) --> Projected innovation ensemble

        if self.keys_da.get('emp_cov', False):
            E_anom = self.solve(scy, enE @ PI)              # shape: (nd, ne)
            invSr = (1/Sr)[:, None]                         # shape: (nr, 1)
            X0 = invSr * (Ur.T @ E_anom)                    # shape: (nr, ne)
            eigval, eigvec = np.linalg.eig(X0 @ X0.T)       # shape: (nr, nr), (nr, nr)
            d = (self.lam + 1) * eigval + 1                 # shape: (nr, )
            rhs = eigvec.T @ (invSr * X1)                   # shape: (nr, ne)
            X2 = invSr * (eigvec @ self.solve(d, rhs))      # shape: (nr, ne)
        else:
            X2 = self.solve(1 + self.lam + Sr**2, X1)       # shape: (nr, ne)

        # AUTO-ADAPTIVE LOCALIZATION
        if self.localization.name == 'autoadaloc':

            if self.localization.projection == 'rank-r':
                Y_anom_proj = np.diag(Sr) @ VrT             # shape: (nr, ne) --> Y_proj = U.T @ Y_anom
                Cxy_loc = self.localization(                # shape: (nx, nr) --> nr < ne << ny (typically)
                    X=scx[:, None]*X_anom,                  # shape: (nx, ne)
                    Y=Y_anom_proj
                )
                return Cxy_loc @ X2                         # shape: (nx, ne)
            
            elif self.localization.projection == 'ensemble':
                Y_anom_proj = X2 @ D_anom                   # shape: (ne, ne)
                return self.localization(
                    X=scx[:, None]*X_anom,                  # shape: (nx, ne)
                    Y=Y_anom_proj                           # shape: (ne, ne)
                )
           
        # DISTANCE-BASED LOCALIZATION
        elif self.localization.name == 'distance_loc':

            # Matrix X: (ne, nd)
            if self.keys_da.get('emp_cov', False):
                X_anom = X_anom * np.sqrt(ne - 1)           # Undo 1/sqrt(ne-1) normalisation
                X = (VrT.T @ eigvec) @ self.solve(d, eigvec.T @ (invSr * Ur.T))
            else:
                X = VrT.T @ (Sr[:, None] * self.solve(1 + self.lam + Sr**2, Ur.T))

            T_loc = self.localization()                     # shape: (nx, nd) --> Localisation mask
            K_loc = T_loc * (scx[:, None] * X_anom @ X)     # shape: (nx, nd) --> Localized gain matrix
            return K_loc @ D_anom                           # shape: (nx, ne)

        # LOCAL ANALYSIS
        elif self.localization.name == 'localanalysis':
            # NOT IMPLEMENTED YET AFTER REFACTORING
            warnings.warn(
                "Local analysis is not currently implemented."
            )
            # TODO: Implement local analysis
            pass

        # PARALLEL UPDATE
        elif self.localization.name == 'parallel_update':
            # NOT IMPLEMENTED YET AFTER REFACTORING
            warnings.warn(
                "Parallel update is not currently implemented."
            )
            # TODO: Implement parallel update
            pass

        # NO LOCALIZATION
        else:
            X3 = VrT.T @ np.diag(Sr) @ X2                   # shape: (ne, ne)
            return scx[:, None] * X_anom @ X3               # shape: (nx, ne)
    

    def solve(self, A, B):
        if A.ndim == 2:
            return solve(A, B)
        else:
            return (A ** (-1))[:, None] * B

    def sqrtm(self, A):
        if A.ndim == 2:
            return sqrtm(A)
        else:
            return np.sqrt(A)

