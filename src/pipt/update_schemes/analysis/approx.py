"""EnRML (IES) without the prior increment term."""

import numpy as np
import warnings

from pipt.update_schemes.analysis.base import AnalysisBase
import pipt.misc_tools.analysis_tools as at


class approx_update(AnalysisBase):
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
        scheme = self.scheme

        # Shapes
        nx, ne = enX.shape
        ny, _  = enY.shape

        # Scaling factors and other attributes needed for the update
        cov = getattr(scheme, 'cov_data', np.eye(ny))  # Data covariance matrix (ny,ny) or (ny,)
        scx = getattr(scheme, 'scale_state', np.ones(nx))
        scy = getattr(scheme, 'scale_data', self.sqrtm(cov))
        PI  = getattr(
            scheme, 'proj',
            (np.eye(ne) - np.ones((ne, ne)) / ne)/ np.sqrt(ne-1)
        )  # shape: (ne, ne) such that A@PI = A - mean(A)/sqrt(ne-1) for any ensemble matrix A of shape (na, ne)

        # Check for adjoint-based update
        if kwargs.get('enAdj', None) is not None:
            Y = kwargs['enAdj'].mean(axis=-1) @ enX @ PI    # shape: (nd, ne)
        else:
            Y = enY @ PI                                    # shape: (nd, ne) --> Such that Cyy ≈ Y @ Y.T

        # Anomaly matrices
        X_anom = self.solve(scx, enX @ PI)                  # shape: (nx, ne) --> State anomalies: (X-mean(X))/sqrt(ne-1)
        Y_anom = self.solve(scy, Y)                         # shape: (nd, ne) --> Predicted data anomalies: (Y-mean(Y))/sqrt(ne-1)
        D_anom = self.solve(scy, enE - enY)                 # shape: (nd, ne) --> Innovation ensemble: data - predictions

        # Truncated SVD on predicted data anomalies
        Ur, Sr, VrT = at.truncSVD(Y_anom, energy=scheme.trunc_energy) # shape: (nd, nr), (nr,), (nr, ne)

        # ===============================================
        # Compute step
        # ===============================================
        X1 = Ur.T @ D_anom                                  # shape: (nr, ne) --> Projected innovation ensemble

        if scheme.keys_da.get('emp_cov', False):
            E_anom = self.solve(scy, enE @ PI)              # shape: (nd, ne)
            invSr = (1/Sr)[:, None]                         # shape: (nr, 1)
            X0 = invSr * (Ur.T @ E_anom)                    # shape: (nr, ne)
            eigval, eigvec = np.linalg.eig(X0 @ X0.T)       # shape: (nr, nr), (nr, nr)
            d = (scheme.lam + 1) * eigval + 1                 # shape: (nr, )
            rhs = eigvec.T @ (invSr * X1)                   # shape: (nr, ne)
            X2 = invSr * (eigvec @ self.solve(d, rhs))      # shape: (nr, ne)
        else:
            X2 = self.solve(1 + scheme.lam + Sr**2, X1)       # shape: (nr, ne)

        # AUTO-ADAPTIVE LOCALIZATION
        localization = scheme.localization
        if localization.name == 'autoadaloc':
            y_proj = localization.info.get('projection', 'rank-r')
            assert y_proj in ['rank-r', 'ensemble'], "Projection method must be either 'rank-r' or 'ensemble'."

            if y_proj == 'rank-r':
                Y_anom_proj = np.diag(Sr) @ VrT             # shape: (nr, ne) --> Y_proj = U.T @ Y_anom
                T_loc = localization(                       # shape: (nx, nr) --> nr < ne << ny (typically)
                    X = scx[:, None]*X_anom,                # shape: (nx, ne)
                    Y = Y_anom_proj
                )
                Cxy_loc = T_loc * (scx[:, None]*X_anom @ Y_anom_proj.T)
                return Cxy_loc @ X2                         # shape: (nx, ne)

            elif y_proj == 'ensemble':
                Y_anom_proj = X2 @ D_anom                   # shape: (ne, ne)
                T_loc = localization(                       # shape: (nx, ne)
                    X = scx[:, None]*X_anom,                # shape: (nx, ne)
                    Y = Y_anom_proj
                )
                step = (T_loc * scx[:, None]*X_anom) @ Y_anom_proj
                return step                                 # shape: (nx, ne)

        # DISTANCE-BASED LOCALIZATION
        elif localization.name == 'distance_loc':

            # Gain-factor matrix X shape: (nr, nd)
            if scheme.keys_da.get('emp_cov', False):
                A = X_anom * np.sqrt(ne - 1)                # Undo 1/sqrt(ne-1) normalisation; shape: (nx, ne)
                X = (VrT.T @ eigvec) @ self.solve(d, eigvec.T @ (invSr * Ur.T))
            else:
                A = scx[:, None] * X_anom                   # shape: (nx, ne)
                X = VrT.T @ (Sr[:, None] * self.solve(1 + scheme.lam + Sr**2, Ur.T))

            T_loc = localization()                          # shape: (nx, nd) -- sparse localisation mask
            K_loc = T_loc.multiply(A @ X)                   # shape: (nx, nd) -- elementwise sparse × dense
            return K_loc @ D_anom                           # shape: (nx, ne)

        # LOCAL ANALYSIS
        elif localization.name == 'localanalysis':
            # NOT IMPLEMENTED YET AFTER REFACTORING
            warnings.warn(
                "Local analysis is not currently implemented."
            )
            # TODO: Implement local analysis
            pass

        # PARALLEL UPDATE
        elif localization.name == 'parallel_update':
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
