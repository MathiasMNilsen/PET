"""Adaptive localization implementation."""
import numpy as np
from typing import Union
from scipy.special import expit
from pipt.localization.common import (
    LocalizationBase,
)

__all__ = ["AutoAdaptiveLocalization"]

class AutoAdaptiveLocalization(LocalizationBase):
    """Adaptive localization strategy and engine implementation."""

    name = "autoadaloc"

    def __init__(self, info: Union[dict, list]):
        """
        Initialize the AutoAdaptiveLocalization instance.

        All configuration is supplied through the ``info`` dictionary, which maps
        directly to a ``[dataassim.localization]`` table in a TOML config file.

        Parameters
        ----------
        info : dict or list
            Localization configuration. Recognised keys:

            **field** : list of int, *required*
                Grid dimensions. For a 3-D reservoir use ``[nz, nx, ny]``;
                for a 2-D field ``[nx, ny]`` is sufficient. Only the product
                (total cell count) is used by this class.

            **actnum** : str, *optional*
                Path to a ``.npz`` file whose first array is a boolean mask
                of active cells. When supplied, only active cells are counted
                toward ``default_num_active``. Default: ``None`` (all cells
                are considered active).

            **threshold** : {``"fixed"``, ``"universal"``, *other*}, *optional*
                Method used to compute the correlation threshold below which
                a correlation is deemed indistinguishable from sampling noise:

                - ``"fixed"`` — threshold equals ``nstd`` directly; no noise
                  estimation is performed. Use when you want a deterministic,
                  reproducible cut-off independent of the ensemble.
                - ``"universal"`` — threshold = ``sqrt(2 * log(N)) * sigma``,
                  where *sigma* is estimated column-wise from shuffled
                  correlations via the MAD estimator. Adapts automatically
                  to ensemble size.
                - *any other string* — threshold = ``nstd * sigma``; a
                  user-controlled multiple of the estimated noise level.

                Default: ``"fixed"``.

            **nstd** : float, *optional*
                Threshold value or noise multiplier (interpretation depends on
                ``threshold``). Larger values suppress more correlations.
                Default: ``1``.

            **type** : {``"hard"``, ``"soft"``, ``"sigm"``}, *optional*
                Tapering strategy applied once the threshold is known:

                - ``"hard"`` — binary mask: 1 where |r| ≥ threshold, 0
                  elsewhere. Sharp cut-off, computationally efficient.
                - ``"soft"`` — smooth rational-function taper that transitions
                  gradually around the threshold. Avoids discontinuities in
                  the localization operator.
                - ``"sigm"`` — sigmoid-based taper; similar smoothness to
                  ``"soft"`` but with a different shape near the transition.

                Default: ``"hard"``.

            **projection** : {``"rank-r"``, ``"ensemble"``}, *optional*
                Method used to assemble the localized cross-covariance:

                - ``"rank-r"`` — ``taper * (X @ Y.T)``. The full
                  (n_state × n_obs) cross-covariance is formed first and
                  then masked element-wise. Standard choice.
                - ``"ensemble"`` — ``(taper * X) @ Y``. The taper is applied
                  directly to the state anomaly columns before projection,
                  avoiding the formation of the full cross-covariance matrix.
                  Preferred for very large state vectors.

                Default: ``"rank-r"``.

        Examples
        --------
        Minimal TOML block inside ``[dataassim]`` using fixed thresholding:

        ```toml
        [dataassim.localization]
        name       = "autoadaloc"
        field      = [1, 20, 20]   # [nz, nx, ny]
        threshold  = "fixed"
        nstd       = 0.4
        type       = "hard"
        projection = "rank-r"
        ```

        Noise-adaptive thresholding with a smooth taper:

        ```toml
        [dataassim.localization]
        name       = "autoadaloc"
        field      = [2, 30, 40]   # two-layer, 30×40 lateral grid
        actnum     = "active_cells.npz"
        threshold  = "universal"   # adapts to ensemble size automatically
        type       = "soft"
        projection = "rank-r"
        ```

        Large state vector — skip forming the full cross-covariance:

        ```toml
        [dataassim.localization]
        name       = "autoadaloc"
        field      = [5, 100, 100]
        threshold  = "fixed"
        nstd       = 0.3
        type       = "hard"
        projection = "ensemble"    # avoids 50000×n_obs dense matrix
        ```
        """
        self.field, self.actnum = self.config_common(info)
        self.nstd = info.get("nstd", 1)
        self.threshold  = info.get("threshold", "fixed")
        self.tapertype  = info.get("type", "hard")
        self.parameters = info.get("parameters", ['NA'])
        self.projection = info.get("projection", "rank-r")

        # Ensure that the tapering type is valid
        if self.tapertype not in ["hard", "soft", "sigm"]:
            raise ValueError(
                f"Invalid tapering type '{self.tapertype}'. "
                "Supported types are 'hard', 'soft', and 'sigm'."
            )

        # Ensure that the projection method is valid
        if self.projection not in ["rank-r", "ensemble"]:
            raise ValueError(
                f"Invalid projection method '{self.projection}'. "
                "Supported methods are 'rank-r' and 'ensemble'."
            )

    def __call__(
            self, 
            X: np.ndarray, 
            Y: np.ndarray, 
            parameters: list[str]=None, 
            prior_info: dict=None
        ) -> np.ndarray:
        """
        Calculate truncated cross-covariance matrix.

        Parameters
        ----------
        X : ndarray, shape (nx, ne)
            State perturbation ensemble.

        Y : ndarray, shape (ny, ne)
            Projected predicted data ensemble.

        parameters : list[str]
            Ordered list of parameters corresponding to blocks in X.

        prior_info : dict, optional
            Prior information for each parameter. If provided,
            ``prior_info[param]["active"]`` specifies the number of
            active variables associated with the parameter.

        Returns
        -------
        ndarray, shape (nx, ny)
            Adaptively localized cross-covariance matrix.
        """
        parameters = self.parameters if parameters is None else parameters
        prior_info = {} if prior_info is None else prior_info

        corr = self.corr_matrix(X, Y) # Shape: (nx, ny)
        corr_shuffled = self.corr_matrix(
            X[:, np.random.permutation(X.shape[1])],
            Y,
        )

        default_num_active = (
            np.sum(self.actnum) if (self.actnum is not None) else np.prod(self.field)
        )

        taper = np.ones_like(corr)
        row_start = 0
        for param in parameters:

            if param == "NA":
                num_active = taper.shape[0] - row_start
            else:
                param_info = prior_info.get(param, {})
                num_active = int(param_info.get("active", default_num_active))

            rows = slice(row_start, row_start + num_active)
            taper[rows] = self.tapering_function(
                corr[rows],
                corr_shuffled[rows],
            )
            row_start += num_active
        
        if self.projection == 'rank-r':
            return taper * (X @ Y.T)
        elif self.projection == 'ensemble':
            return (taper * X) @ Y


    def tapering_function(self, corr_values: np.ndarray, corr_values_shuffled: np.ndarray) -> np.ndarray:

        """
        Compute tapering coefficients from sample correlations.

        The tapering coefficients are used to suppress correlations that are
        indistinguishable from noise. A noise level is estimated for each
        observation variable from the corresponding shuffled correlations using
        the median absolute deviation (MAD),

            sigma = median(|r_shuffled|) / 0.6745

        which provides a robust estimate of the standard deviation under the
        assumption of Gaussian noise.

        Depending on the localization settings, the correlation threshold is
        computed using one of the following methods:

        - ``"universal"``:
            threshold = sqrt(2 log(N)) * sigma
        - ``"fixed"``:
            threshold = nstd
        - otherwise:
            threshold = nstd * sigma

        Tapering can then be applied using one of three strategies:

        - ``"hard"`` (default):
            correlations above the threshold are assigned a taper value of 1,
            otherwise 0.
        - ``"soft"``:
            smooth tapering based on ``rational_function``.
        - ``"sigm"``:
            sigmoid-based tapering using ``rational_function_sigmoid``.

        Parameters
        ----------
        corr_values : ndarray of shape (nx, ny)
            Sample correlation matrix.

        corr_values_shuffled : ndarray of shape (nx, ny)
            Correlation matrix computed from shuffled or randomized ensembles.
            Used to estimate the noise level of the correlations.

        Returns
        -------
        ndarray of shape (nx, ny)
            Tapering coefficients in the interval [0, 1]. These coefficients
            can be applied element-wise to the correlation matrix to reduce
            the influence of correlations attributed to sampling noise.
        """
        taper_coeff = np.zeros_like(corr_values)
        for i in range(corr_values.shape[1]):
            corr = corr_values[:, i]

            # Estimate noise level from shuffled correlations: 
            mad_to_std = 1 / 0.6745
            noise_std  = np.median(np.abs(corr_values_shuffled[:, i])) * mad_to_std

            # Compute threshold
            if self.threshold == "universal":
                threshold = np.sqrt(2 * np.log(corr.size)) * noise_std
            elif self.threshold == "fixed":
                threshold = self.nstd
            else:
                threshold = self.nstd * noise_std

            # Compute taper coefficients
            if self.tapertype == "soft":
                taper = self.rational_function(
                    1 - np.abs(corr),
                    1 - threshold,
                )
            elif self.tapertype == "sigm":
                taper = self.rational_function_sigmoid(
                    np.abs(corr),
                    self.nstd,
                )
            else:
                taper = np.zeros_like(corr)
                taper[np.abs(corr) > threshold] = 1.0

            taper_coeff[:, i] = taper

        return taper_coeff


    def rational_function(self, distance, length_scale):
        z_ratio = np.absolute(distance) / length_scale
        idx_inner = np.where(z_ratio <= 1)
        idx_outer = np.where(z_ratio <= 2)
        idx_transition = np.setdiff1d(idx_outer, idx_inner)

        taper = np.zeros(len(z_ratio))

        taper[idx_inner] = (
            1
            - (np.power(z_ratio[idx_inner], 5) / 4)
            + (np.power(z_ratio[idx_inner], 4) / 2)
            + (5 * np.power(z_ratio[idx_inner], 3) / 8)
            - (5 * np.power(z_ratio[idx_inner], 2) / 3)
        )

        taper[idx_transition] = (
            (np.power(z_ratio[idx_transition], 5) / 12)
            - (np.power(z_ratio[idx_transition], 4) / 2)
            + (5 * np.power(z_ratio[idx_transition], 3) / 8)
            + (5 * np.power(z_ratio[idx_transition], 2) / 3)
            - 5 * z_ratio[idx_transition]
            - np.divide(2, 3 * z_ratio[idx_transition])
            + 4
        )

        return taper

    @staticmethod
    def rational_function_sigmoid(distance, length_scale):
        steepness = 50
        return expit((distance - (1 - length_scale)) * steepness)
    
    @staticmethod
    def corr_matrix(X, Y, eps=1e-6):
        """
        Compute the correlation matrix between two ensemble matrices X and Y.

        Parameters
        ----------
        X : np.ndarray, shape (nx, ne)
        Y : np.ndarray, shape (ny, ne)
        eps : float, optional, default=1e-6
            Small value to avoid division by zero when computing standard deviations.

        Returns
        -------
        corr : np.ndarray, shape (nx, ny)
            The correlation matrix between X and Y.
        """
        stdX = np.std(X, axis=1)
        stdY = np.std(Y, axis=1)

        nx = X.shape[0]
        corr = np.corrcoef(X, Y)[:nx, nx:]
        corr[stdX < eps, :] = 0
        corr[:, stdY < eps] = 0

        return np.nan_to_num(corr)
        

