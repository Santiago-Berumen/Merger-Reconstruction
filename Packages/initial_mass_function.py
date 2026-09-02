import numpy as np
from scipy.special import expit
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

import bilby


class TaperedPowerLaw(bilby.core.prior.Prior):
    """
    Power-law prior with a smooth low-mass taper.

    p(m) ∝ m^(-alpha) S(m | mmin, delta_m)

    where
        S = 0                                      for m < mmin
        S = 1 / (f(m-mmin, delta_m) + 1)           for mmin <= m < mmin + delta_m
        S = 1                                      for m >= mmin + delta_m
    and
        f(x, delta_m) =
            exp(delta_m / x + delta_m / (x - delta_m)).

    Parameters
    ----------
    alpha : float
        Power-law spectral index.
    minimum : float
        Minimum allowed mass.
    maximum : float
        Maximum allowed mass.
    delta_m : float
        Width of the low-mass taper.
    """

    def __init__(self, alpha, minimum, maximum, delta_m=0.0, name=None, 
                 latex_label=None,  unit=None, n_grid=10000):
        self.alpha = alpha
        self.delta_m = delta_m
        self.n_grid = n_grid

        if minimum <= 0:
            raise ValueError("minimum must be positive.")

        if maximum <= minimum:
            raise ValueError("maximum must be greater than minimum.")

        if delta_m < 0:
            raise ValueError("delta_m must be non-negative.")

        # Initialize bilby Prior object
        super().__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
            minimum=minimum,
            maximum=maximum,
        )

        # Construct numerical CDF / inverse CDF
        self.setup_cdf()

    @property
    def mass_grid(self):
        """Mass grid on which the pdf/cdf are tabulated."""
        return self._mass_grid

    @property
    def pdf_grid(self):
        """
        Normalized analytic pdf evaluated on `mass_grid`.

        Convenience for plotting, e.g.

            plt.plot(prior.mass_grid, prior.pdf_grid)

        which is identical to ``prior.prob(prior.mass_grid)``.
        """
        return self._pdf_grid

    @property
    def cdf_grid(self):
        """Normalized analytic cdf evaluated on `mass_grid`."""
        return self._cdf_grid

    def taper(self, mass):
        """Smooth taper S(m | mmin, delta_m)."""

        mass = np.asarray(mass)
        S = np.zeros_like(mass, dtype=float)

        # if delta_m is zero, taper is just 1 above mmin 
        if self.delta_m == 0:
            S[mass >= self.minimum] = 1.0
            return S

        # Fully unsuppressed region
        high = mass >= self.minimum + self.delta_m
        S[high] = 1.0

        # Tapering region: x = m - mmin lies in (0, delta_m).
        # x = 0 is excluded because log_f -> +inf there, i.e. S -> 0,
        # which is already the initialized value.
        mid = (
            (mass > self.minimum)
            & (mass < self.minimum + self.delta_m)
        )

        if np.any(mid):
            x = mass[mid] - self.minimum

            # log(f) = delta/x + delta/(x - delta)
            #
            # x -> 0+  : log_f -> +inf  =>  f -> inf  =>  S -> 0
            # x -> delta-: log_f -> -inf =>  f -> 0    =>  S -> 1
            log_f = (
                self.delta_m / x
                + self.delta_m / (x - self.delta_m)
            )

            # S = 1 / (f + 1) = 1 / (1 + exp(log_f)) = expit(-log_f)
            S[mid] = expit(-log_f)

        return S

    def prob(self, mass):
        """Normalized probability density."""

        mass = np.asarray(mass)

        pdf = np.zeros_like(mass, dtype=float)

        valid = (
            (mass >= self.minimum)
            & (mass <= self.maximum)
        )

        if np.any(valid):
            pdf[valid] = (
                mass[valid] ** (-self.alpha)
                * self.taper(mass[valid])
                / self._normalization
            )

        return pdf

    def ln_prob(self, mass):
        """Log probability density."""

        mass = np.asarray(mass)

        result = np.full_like(
            mass,
            -np.inf,
            dtype=float,
        )

        valid = (
            (mass >= self.minimum)
            & (mass <= self.maximum)
        )

        if np.any(valid):
            result[valid] = (
                -self.alpha * np.log(mass[valid])
                + np.log(self.taper(mass[valid]))
                - np.log(self._normalization)
            )

        return result

    def setup_cdf(self):
        """Construct numerical CDF and inverse CDF."""

        # Include the endpoints explicitly
        self._mass_grid = np.linspace(
            self.minimum,
            self.maximum,
            self.n_grid,
        )

        unnormalized_pdf = (
            self._mass_grid ** (-self.alpha)
            * self.taper(self._mass_grid)
        )

        # Numerical normalization
        self._normalization = np.trapz(
            unnormalized_pdf,
            self._mass_grid,
        )

        # Analytic (normalized) pdf on the grid, for plotting
        self._pdf_grid = unnormalized_pdf / self._normalization

        # Numerical CDF
        cdf = cumulative_trapezoid(
            unnormalized_pdf,
            self._mass_grid,
            initial=0,
        )

        cdf /= cdf[-1]

        self._cdf_grid = cdf

        # Remove duplicate CDF values that can occur because
        # the taper is numerically zero near mmin.
        unique_cdf, indices = np.unique(
            cdf,
            return_index=True,
        )

        self._inverse_cdf = interp1d(
            unique_cdf,
            self._mass_grid[indices],
            bounds_error=False,
            fill_value=(
                self.minimum,
                self.maximum,
            ),
        )

    def cdf(self, mass):
        """Numerical cumulative distribution function."""

        mass = np.asarray(mass)

        result = np.zeros_like(mass, dtype=float)

        result[mass >= self.maximum] = 1.0

        valid = (
            (mass > self.minimum)
            & (mass < self.maximum)
        )

        if np.any(valid):
            result[valid] = np.interp(
                mass[valid],
                self._mass_grid,
                self._cdf_grid,
            )

        return result

    def rescale(self, val):
        """
        Transform uniform samples u in [0, 1] into samples
        from this prior using the inverse CDF.
        """

        val = np.asarray(val)

        if np.any((val < 0) | (val > 1)):
            raise ValueError("rescale values must be in [0, 1].")

        return self._inverse_cdf(val)