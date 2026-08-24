"""Full (model-space) LM ensemble update."""

import numpy as np

from pipt.update_schemes.analysis.base import AnalysisBase
import pipt.misc_tools.analysis_tools as at


class full_update(AnalysisBase):
    """
    Full LM update as in Chen & Oliver (2013).

    Unlike the approximate update, the state-error covariance is represented
    in model space via the ``Am`` matrix, which adds an explicit regularisation
    term pulling the ensemble toward the prior.

    Reference
    ---------
    Chen, Y., & Oliver, D. S. (2013). Levenberg-Marquardt forms of the iterative
    ensemble smoother for efficient history matching and uncertainty quantification.
    Computational Geosciences, 17(4), 689-703.
    https://doi.org/10.1007/s10596-013-9351-5

    Note
    ----
    No localization is implemented for this update scheme.
    """

    def update(self, enX, enY, enE, **kwargs):
        """
        Perform the full LM update.

        Parameters
        ----------
        enX : np.ndarray, shape (nx, ne)
            State ensemble matrix.
        enY : np.ndarray, shape (nd, ne)
            Predicted data ensemble matrix.
        enE : np.ndarray, shape (nd, ne)
            Perturbed observations ensemble.

        Returns
        -------
        np.ndarray, shape (nx, ne)
            Update step to be added to the state ensemble.
        """
        scheme = self.scheme

        nx, ne = enX.shape
        ny, _  = enY.shape

        # Scaling factors and projection matrix
        cov    = getattr(scheme, 'cov_data', np.eye(ny))
        scx    = getattr(scheme, 'scale_state', np.ones(nx))
        scy    = getattr(scheme, 'scale_data', self.sqrtm(cov))
        PI     = getattr(scheme, 'proj',
                         (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1))

        priorX = kwargs.get('prior', scheme.prior_enX)

        # Build Am matrix once per outer iteration
        if scheme.Am is None:
            self.ext_Am()

        # Anomaly matrices
        Y_anom = self.solve(scy, enY @ PI)              # shape: (nd, ne)
        X_anom = self.solve(scx, enX @ PI)              # shape: (nx, ne)
        D_anom = self.solve(scy, enE - enY)             # shape: (nd, ne)

        # Truncated SVD of predicted-data anomalies
        Ur, Sr, VrT = at.truncSVD(Y_anom, energy=scheme.trunc_energy)  # (nd,nr), (nr,), (nr,ne)

        # ── Data-misfit term (δm₁) ──────────────────────────────────────────
        X1 = Ur.T @ D_anom                              # shape: (nr, ne)
        X2 = self.solve(1 + scheme.lam + Sr ** 2, X1)   # shape: (nr, ne)
        X3 = VrT.T @ np.diag(Sr) @ X2                   # shape: (ne, ne)
        delta_m1 = (scx[:, None] * X_anom) @ X3         # shape: (nx, ne)

        # ── Regularisation term (δm₂) -- model-space prior pull ─────────────
        Am = scheme.Am
        X4 = Am.T @ self.solve(scx, enX - priorX)       # shape: (nr', ne)
        X5 = Am @ X4                                    # shape: (nx,  ne)
        X6 = X_anom.T @ X5                              # shape: (ne,  ne)
        X7 = VrT.T @ self.solve(1 + scheme.lam + Sr ** 2,
                                VrT @ X6)               # shape: (ne, ne)
        delta_m2 = -(scx[:, None] * X_anom) @ X7        # shape: (nx, ne)

        return delta_m1 + delta_m2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def ext_Am(self):
        """Compute and cache the Am matrix from the scaled prior ensemble."""
        scheme = self.scheme
        delta = scheme.state_scaling[:, None] * (scheme.prior_enX @ scheme.proj)
        U, S, _ = np.linalg.svd(delta, full_matrices=False)

        # Truncate to the energy threshold
        r = int(np.searchsorted(np.cumsum(S) / S.sum(), self.scheme.trunc_energy)) + 1
        scheme.Am = U[:, :r] * (S[:r] ** (-1))[None, :]   # shape: (nx, r), notation from paper
