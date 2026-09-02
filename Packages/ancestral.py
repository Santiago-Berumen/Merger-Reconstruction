#!/usr/bin/env python
# coding: utf-8
# system functions that are always useful to have
import os
import json
import inspect
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# A blanket DeprecationWarning filter used to sit here. It hid NumPy's
# "conversion of an array with ndim > 0 to a scalar is deprecated" warning, which in
# NumPy 2 became a hard error and broke every likelihood call. Deprecations are left
# visible so the next one surfaces before it becomes a breakage.

import numpy as np
import scipy
import bilby

# AncestralLikelihood2D reads self.parameters inside log_likelihood, which raises 
# a FutureWarning in bilby, so let's silence that
bilby.core.likelihood.set_parameters_as_state("TRUE")
warnings.filterwarnings(
    "ignore",
    message=r".*does not accept 'parameters' as an argument",
    category=FutureWarning,
)

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import tqdm
from p_tqdm import p_map

LOG_2PI = np.log(2.0 * np.pi)


def as_float(value):
    """
    Convert a scalar-like value to a Python float.

    NR fits and the scipy/sklearn density estimators return shape-(1,) arrays. NumPy 2
    removed the implicit float(size-1 array) conversion, so go through .item(), which
    accepts 0-d arrays, size-1 arrays and plain Python floats -- and raises on anything
    with more than one element rather than silently taking the first.
    """
    return np.asarray(value).item()


def gmm_fast_params(estimator):
    """
    Precompute the per-component arrays that GaussianMixture.score_samples rebuilds
    on every call, so a single-point density evaluation is a few numpy ops instead of
    a full sklearn estimator call (input validation, fitted checks, array allocation).

    Returns None when the estimator is not a plain GaussianMixture -- notably
    BayesianGaussianMixture ('dpgmm'), whose log-density carries extra
    degrees-of-freedom and digamma terms -- in which case callers must fall back to
    estimator.score_samples.
    """
    if not isinstance(estimator, GaussianMixture):
        return None

    covariance_type = estimator.covariance_type
    means = estimator.means_
    prec_chol = estimator.precisions_cholesky_
    n_features = means.shape[1]

    # log|L| per component and the precomputed mean.L product, matching
    # sklearn's _compute_log_det_cholesky / _estimate_log_gaussian_prob
    if covariance_type == 'full':
        log_det = np.sum(np.log(np.diagonal(prec_chol, axis1=1, axis2=2)), axis=1)
        shift = np.einsum('kd,kde->ke', means, prec_chol)
    elif covariance_type == 'tied':
        log_det = np.full(means.shape[0], np.sum(np.log(np.diag(prec_chol))))
        shift = means @ prec_chol
    elif covariance_type == 'diag':
        log_det = np.sum(np.log(prec_chol), axis=1)
        shift = means * prec_chol
    elif covariance_type == 'spherical':
        log_det = n_features * np.log(prec_chol)
        shift = means * prec_chol[:, np.newaxis]
    else:
        return None

    return {
        'covariance_type': covariance_type,
        'prec_chol': prec_chol,
        'shift': shift,
        'log_det': log_det,
        'log_weights': np.log(estimator.weights_),
        'n_features': n_features,
    }


def gmm_fast_logpdf(fast, x):
    """Log-density of the mixture at a single point x, shape (n_features,)."""
    covariance_type = fast['covariance_type']
    prec_chol = fast['prec_chol']

    if covariance_type == 'full':
        y = np.einsum('d,kde->ke', x, prec_chol) - fast['shift']
    elif covariance_type == 'tied':
        y = (x @ prec_chol) - fast['shift']
    elif covariance_type == 'diag':
        y = x * prec_chol - fast['shift']
    else:  # spherical
        y = x * prec_chol[:, np.newaxis] - fast['shift']

    mahalanobis = np.einsum('ke,ke->k', y, y)
    log_prob = (
        -0.5 * (fast['n_features'] * LOG_2PI + mahalanobis)
        + fast['log_det']
        + fast['log_weights']
    )
    # logsumexp over components
    peak = log_prob.max()
    return peak + np.log(np.exp(log_prob - peak).sum())


class AncestralLikelihood2D(bilby.core.likelihood.Likelihood):
    """
    A Class for Computing Likelihood to Track Ancestral Compact Objects of Hierarchical Merger Candidate BH.

    It supports four methods for estimating 2D probability distributions:
    - 'hist': Histogram-based
    - 'kde': Gaussian KDE (via scipy.stats)
    - 'gmm': Gaussian Mixture Model (via sklearn)
    - 'dpgmm': Dirichlet Process GMM (via sklearn)

    The class provides two interpolation methods for evaluating the likelihood using the 'hist' method (other methods do not need any interpolation):
    - 'interp2d': accepted for backwards compatibility; also uses RegularGridInterpolator
    - 'regular_grid_intrp': RegularGridInterpolator (via scipy.interpolate)

    Note: scipy.interpolate.interp2d was removed in SciPy 1.14, so both options now
    build a RegularGridInterpolator and behave identically.

    Parameters
    ----------
    parameter_names: dict
        Dictionary of parameter names with keys as parameter names.
    mass_pos: array_like
        Posterior samples of the observed mass.
    spin_pos: array_like
        Posterior samples of the observed spin.
    mass_pr: array_like or None, optional
        Prior samples of the observed mass (default: None).
    spin_pr: array_like or None, optional
        Prior samples of the observed spin (default: None).
    method: str, optional
        Method for computing 2D distributions. Options are:
        - 'hist': Uses histogram binning.
        - 'kde': Uses scipy.stats.gaussian_kde for kernel density estimation.
        - 'gmm': Uses sklearn.mixture.GaussianMixture for kernel density estimation.
        - 'dpgmm': Uses sklearn.mixture.BayesianGaussianMixture for kernel density estimation.
        Default is 'kde'.
    method_kwargs: dict, optional
        Additional arguments for the chosen method. For example:
        - If method='hist', supports {'bins': int} to define histogram resolution.
        - If method='kde', supports {'bandwidth': float} to define bandwidth of kde.
        - If method='gmm', supports {'n_components': int, 'covariance_type': str, 'max_iter': int, 'tol': float, 'random_state': int}
        - If method='dpgmm', supports {'n_components': int, 'covariance_type': str, 'max_iter': int, 'tol': float, 'random_state': int}
        Default is an empty dictionary.
    interp_method: str, optional
        Interpolation method for computing likelihood using the 'hist' method. Both
        options now use scipy.interpolate.RegularGridInterpolator; 'interp2d' is kept
        as an accepted value because scipy removed interp2d in 1.14.
        Default is 'interp2d'.
    Mfin_NRfit: callable
        Function for final mass fitting, which takes model parameters as input and returns a mass value.
    Chifin_NRfit: callable
        Function for final spin fitting, which takes model parameters as input and returns a spin value.
    """
    def __init__(self, parameter_names, mass_pos, spin_pos, mass_pr=None, spin_pr=None, method='kde',
                 method_kwargs=None, interp_method='interp2d', Mfin_NRfit=None, Chifin_NRfit=None):

        super().__init__(parameters={param: None for param in parameter_names})
        self.mass_pos = np.asarray(mass_pos)
        self.spin_pos = np.asarray(spin_pos)
        self.mass_pr = np.asarray(mass_pr) if mass_pr is not None else None
        self.spin_pr = np.asarray(spin_pr) if spin_pr is not None else None
        self.method = method.lower()
        self.interp_method = interp_method.lower()
        self.method_kwargs = method_kwargs if method_kwargs is not None else {}
        self.parameter_names = parameter_names

        # Validate up front. Otherwise a typo surfaces as a ValueError from inside the
        # first likelihood call, i.e. after the sampler has already started.
        if self.method not in ('hist', 'kde', 'gmm', 'dpgmm'):
            raise ValueError(
                f"Invalid method {method!r}. Choose 'hist', 'kde', 'gmm', or 'dpgmm'."
            )
        if self.interp_method not in ('interp2d', 'regular_grid_intrp'):
            raise ValueError(
                f"Invalid interpolation method {interp_method!r}. "
                "Choose 'interp2d' or 'regular_grid_intrp'."
            )

        if not callable(Mfin_NRfit) or not callable(Chifin_NRfit):
            raise ValueError("Mfin_NRfit and Chifin_NRfit must be callable functions.")

        self.Mfin_NRfit = Mfin_NRfit
        self.Chifin_NRfit = Chifin_NRfit

        # Combine posterior and prior samples to define the bin ranges
        combined_mass_samples = np.append(self.mass_pos, self.mass_pr) if self.mass_pr is not None else self.mass_pos
        combined_spin_samples = np.append(self.spin_pos, self.spin_pr) if self.spin_pr is not None else self.spin_pos

        # Default bins setup  (Needed for 'hist' method)
        bins = self.method_kwargs.get('bins', 65)
        self.mass_bins = np.linspace(np.min(combined_mass_samples), np.max(combined_mass_samples), bins)
        self.spin_bins = np.linspace(np.min(combined_spin_samples), np.max(combined_spin_samples), bins)
        self.mass_midpoints = (self.mass_bins[:-1] + self.mass_bins[1:]) / 2
        self.spin_midpoints = (self.spin_bins[:-1] + self.spin_bins[1:]) / 2
        self.interp_obj = None

        # Cache KDE to avoid recomputation in 'kde', 'gmm', and 'dpgmm' methods
        self.cached_pos_kde = None
        self.cached_prior_kde = None

    def compute_2d_distribution(self, mass_samples, spin_samples, method):
        """
        Compute the 2D distribution using the selected method ('hist', 'kde' with caching, 'gmm' with caching, or 'dpgmm' with caching).

        Parameters:
            mass_samples (array-like): Samples for the mass.
            spin_samples (array-like): Samples for the spin.
            method (str): The method to compute the distribution ('hist', 'kde', 'gmm', or 'dpgmm').

        Returns:
            In 'hist' method
                pdf_2d (array): The computed 2D probability density function.
                axes (tuple): Midpoints of mass and spin.
            In 'kde', 'gmm', or 'dpgmm' methods
                cached_pos_kde and cached_prior_kde
        """
        if method == 'hist':
            pdf_2d, _, _ = np.histogram2d(mass_samples, spin_samples, bins=(self.mass_bins, self.spin_bins), density=True)
            pdf_2d = pdf_2d.T # Transpose for correct orientation
        elif method == 'kde':
            kde = scipy.stats.gaussian_kde(np.vstack([mass_samples, spin_samples]), bw_method=self.method_kwargs.get('bandwidth', 0.10))
            cached_result = {'kde': kde}
            if mass_samples is self.mass_pos and spin_samples is self.spin_pos:
                self.cached_pos_kde = cached_result
            if mass_samples is self.mass_pr and spin_samples is self.spin_pr:
                self.cached_prior_kde = cached_result
            return cached_result
        elif method in ('gmm', 'dpgmm'):
            samples = np.vstack((mass_samples, spin_samples)).T
            if method == 'gmm':
                estimator = GaussianMixture(
                    n_components=self.method_kwargs.get("n_components", 5),
                    covariance_type=self.method_kwargs.get("covariance_type", "full"),
                    max_iter=self.method_kwargs.get("max_iter", 1000),
                    tol=self.method_kwargs.get("tol", 1e-3),
                    random_state=self.method_kwargs.get("random_state", 42)
                )
            else:
                estimator = BayesianGaussianMixture(
                    n_components=self.method_kwargs.get("n_components", 20),
                    covariance_type=self.method_kwargs.get("covariance_type", "full"),
                    max_iter=self.method_kwargs.get("max_iter", 1000),
                    tol=self.method_kwargs.get("tol", 1e-3),
                    random_state=self.method_kwargs.get("random_state", 42)
                )
            estimator.fit(samples)
            # 'fast' is None for dpgmm / unsupported covariance types; callers fall back
            cached_result = {'model': estimator, 'fast': gmm_fast_params(estimator)}
            if mass_samples is self.mass_pos and spin_samples is self.spin_pos:
                self.cached_pos_kde = cached_result
            if mass_samples is self.mass_pr and spin_samples is self.spin_pr:
                self.cached_prior_kde = cached_result
            return cached_result
        else:
            raise ValueError("Invalid method. Choose 'hist', 'kde', 'gmm', or 'dpgmm'.")
        return pdf_2d, (self.mass_midpoints, self.spin_midpoints)

    def evaluate_pdf_at_point(self, m, chi, cache):
        """Directly evaluate kde/gmm/dpgmm at (m, chi)."""
        if self.method == 'kde':
            return as_float(cache['kde']([[m], [chi]]))
        elif self.method in ('gmm', 'dpgmm'):
            fast = cache.get('fast')
            if fast is not None:
                return as_float(np.exp(gmm_fast_logpdf(fast, np.array([m, chi]))))
            return as_float(np.exp(cache['model'].score_samples(np.array([[m, chi]]))))
        else:
            raise ValueError("evaluate_pdf_at_point called for method that requires interpolation.")

    def build_hist_interpolator(self):
        """
        Build the interpolator over the (posterior / prior) histogram ratio.

        Both interp_method options now return a RegularGridInterpolator.
        scipy.interpolate.interp2d was removed in SciPy 1.14 -- it survives only as a
        stub that raises NotImplementedError -- and RegularGridInterpolator is the
        replacement scipy itself recommends for regular grids. 'interp2d' is kept as
        an accepted value so existing call sites keep working.
        """
        if self.interp_method not in ('interp2d', 'regular_grid_intrp'):
            raise ValueError("Invalid interpolation method. Choose 'interp2d' or 'regular_grid_intrp'.")

        pos_pdf_2d, _ = self.compute_2d_distribution(self.mass_pos, self.spin_pos, 'hist')

        # Compute the 2D prior distribution if prior samples are provided
        if self.mass_pr is not None and self.spin_pr is not None:
            prior_pdf_2d, _ = self.compute_2d_distribution(self.mass_pr, self.spin_pr, 'hist')
            # Compute the posterior/prior ratio
            pos_prior_pdf_ratio = np.divide(pos_pdf_2d, prior_pdf_2d + 1e-300, out=np.zeros_like(pos_pdf_2d), where=prior_pdf_2d != 0)
            # Avoid zeros, NaNs and infinities by replacing them with a small value
            pos_prior_pdf_ratio[pos_prior_pdf_ratio <= 0] = 1e-300
            pos_prior_pdf_ratio[~np.isfinite(pos_prior_pdf_ratio)] = 1e-300
        else:
            pos_prior_pdf_ratio = pos_pdf_2d

        # pos_prior_pdf_ratio is indexed [spin, mass] but RegularGridInterpolator wants
        # values indexed in the same order as the grids it is given, i.e. [mass, spin], 
        # hence the .T here
        return scipy.interpolate.RegularGridInterpolator(
            (self.mass_midpoints, self.spin_midpoints),
            pos_prior_pdf_ratio.T,
            bounds_error=False,
            fill_value=0.0,
        )

    def log_likelihood(self):
        """Compute the log-likelihood for hierarchical merger candidate BH tracking."""
        # as_float, not float: NR fits commonly return shape-(1,) arrays, which NumPy 2
        # refuses to convert implicitly.
        Mfit_value = as_float(self.Mfin_NRfit(**self.parameters))
        Chifit_value = as_float(self.Chifin_NRfit(**self.parameters))
        
        if (
            Mfit_value <= 0 or Chifit_value < 0 or Chifit_value >= 1
            or not np.isfinite(Mfit_value) or not np.isfinite(Chifit_value)
        ):
            return -np.inf

        if self.method == 'hist':
            # The histogram and its interpolator depend only on the (fixed) samples,
            # so build them once and reuse. Rebuilding them per call made the 'hist'
            # method orders of magnitude slower than the density-estimator methods.
            if self.interp_obj is None:
                self.interp_obj = self.build_hist_interpolator()
            return as_float(np.log(max(
                as_float(self.interp_obj((Mfit_value, Chifit_value))), 1e-300
            )))

        else:
            pos_cache = self.cached_pos_kde or self.compute_2d_distribution(self.mass_pos, self.spin_pos, self.method)
            pos_val = self.evaluate_pdf_at_point(Mfit_value, Chifit_value, pos_cache)

            if self.mass_pr is not None and self.spin_pr is not None:
                prior_cache = self.cached_prior_kde or self.compute_2d_distribution(self.mass_pr, self.spin_pr, self.method)
                prior_val = self.evaluate_pdf_at_point(Mfit_value, Chifit_value, prior_cache)
                pos_prior_pdf_ratio = pos_val / (prior_val + 1e-300)
                return as_float(np.log(max(pos_prior_pdf_ratio, 1e-300)))
            else:
                return as_float(np.log(max(pos_val, 1e-300)))


class Genealogy_Reconstruction:
    """
    A Class to Track Ancestral Compact Objects of Hierarchical Merger Candidate BH
    """
    def __init__(
        self,
        posteriors_candidate,
        parameter_names,
        priors,
        label,
        outdir,
        Mfin_NRfit,
        Chifin_NRfit,
        Vkick_NRfit,
        Chieff,
        chip,
        method="kde",
        priors_candidate=None,
        method_kwargs=None,
        interp_method='interp2d',
        sampler='dynesty',
        sampler_kwargs=None
    ):

        """
        Initializes the class with likelihood estimation and PCA functionality.

        Parameters:
        - posteriors_candidate (dict): Posterior samples of the candidate binary.
        - parameter_names (list): Names of parameters.
        - priors (bilby.prior.PriorDict): Priors for the hyperparameter inference.
        - label (str): Label for output files/Name identifier for the run.
        - outdir (str): Output directory for storing results.
        - Mfin_NRfit (func): callable
        Function for final mass fitting, which takes model parameters as input and returns a mass value.
        - Chifin_NRfit (func): callable
        Function for final spin fitting, which takes model parameters as input and returns a spin value.
        - Vkick_NRfit (func): callable
        Function for final kick fitting, which takes model parameters as input and returns a kick value.
        - Chieff (func): callable
        The effective spin parameter (chi_eff).
        - chip (func): callable
        The effective precession parameter (chi_p).
        - method (str, optional): Density estimation method ('hist', 'kde', 'gmm' or 'dpgmm') (default: 'kde').
        - method_kwargs (dict, optional): Dictionary containing hyperparameters for density estimators.
        - priors_candidate (dict, optional): Priors for the candidate binary.
        - interp_method (str, optional): Interpolation method for computing likelihood using 'hist' method.
        - sampler (str, optional): The inference sampler to use (default: 'dynesty').
        - sampler_kwargs (dict, optional): Additional arguments for the sampler (default: standard Dynesty settings).
        """

        self.posteriors_candidate = posteriors_candidate
        self.priors_candidate = priors_candidate
        self.parameter_names = parameter_names
        self.priors = priors
        self.label = label
        self.outdir = outdir
        self.method = method
        self.method_kwargs = method_kwargs if method_kwargs else {
            # Optional, defaults to a dictionary with common settings
            'bins': 65, #comes with "hist"
            'bandwidth': 0.10, #comes with 'kde'
            "n_components": 5, #comes with "GMM", "DPGMM"
            "covariance_type": "full", #comes with "GMM" and "DPGMM" ['full', 'tied', 'diag', 'spherical'], defaults to "full"
            "max_iter": 1000, #comes with "GMM" and "DPGMM"
            "tol": 1e-3, #comes with "GMM" and "DPGMM"
            "random_state": 42, #comes with "GMM" and "DPGMM"
        }
        self.interp_method = interp_method
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs if sampler_kwargs else {
            "nlive": 1000,
            "naccept": 60,
            "check_point_plot": True,
            "check_point_delta_t": 1800,
            "print_method": 'interval-60',
            "sample": 'acceptance-walk',
            "dlogz": 0.1,
            "npool": 64
        }
        self.Mfin_NRfit = Mfin_NRfit
        self.Chifin_NRfit = Chifin_NRfit
        self.Vkick_NRfit = Vkick_NRfit
        self.Chieff = Chieff
        self.chip = chip
        self.result = None
        self.chieff_p_data = None
        self.chip_p_data = None
        self.vkick_p_data = None
        self.Mf_p_data = None
        self.Chif_p_data = None

        # Create output directory if it doesn't exist
        bilby.utils.check_directory_exists_and_if_not_mkdir(self.outdir)

    def run_analysis(self):
        """
        Run the genealogical reconstruction analysis
        """
        print('Start')

        # Prepare arguments for AncestralLikelihood2D
        for required in ('mass_source', 'spin'):
            if required not in self.posteriors_candidate:
                raise KeyError(
                    f"posteriors_candidate is missing the {required!r} column "
                    f"(has: {sorted(self.posteriors_candidate)})."
                )
        mass_pos = self.posteriors_candidate['mass_source']
        spin_pos = self.posteriors_candidate['spin']

        if self.priors_candidate is not None:
            mass_pr = self.priors_candidate['mass_source']
            spin_pr = self.priors_candidate['spin']
        else:
            mass_pr = None
            spin_pr = None

        # Create instance of AncestralLikelihood2D
        Hyper_likelihood = AncestralLikelihood2D(
            mass_pos=mass_pos,
            spin_pos=spin_pos,
            mass_pr=mass_pr,
            spin_pr=spin_pr,
            method=self.method,
            method_kwargs=self.method_kwargs,
            interp_method=self.interp_method,
            Mfin_NRfit=self.Mfin_NRfit,
            Chifin_NRfit=self.Chifin_NRfit,
            parameter_names=self.parameter_names
        )

        # Run sampler with the defined likelihood and priors
        print('Sampling started for estimating the parameters of parent binary')
        self.result = bilby.run_sampler(
            likelihood=Hyper_likelihood,
            priors=self.priors,
            sampler=self.sampler,
            outdir=self.outdir,
            label=self.label,
            **self.sampler_kwargs
        )
        print('Sampling ended')

        # Plot a corner plot for posterior distributions: all outputs are stored in outdir
        self.result.plot_corner(quantiles=[0.05, 0.95])

        # Marginal Distribution plots
        # self.result.plot_marginals(quantiles=[0.05, 0.95])

        # No plt.show() here: the plots are already written to outdir, and showing them
        # blocks on an interactive backend and warns on a headless one (cluster runs).
        plt.close('all')

        print('Started Chieff/Chip/Vkick/Mfinal/Chifinal calculation')
        # Initialize arrays for the output data
        self.chieff_p_data = np.zeros(len(self.result.posterior), dtype=float)
        self.chip_p_data = np.zeros(len(self.result.posterior), dtype=float)
        self.vkick_p_data = np.zeros(len(self.result.posterior), dtype=float)
        self.Mf_p_data = np.zeros(len(self.result.posterior), dtype=float)
        self.Chif_p_data = np.zeros(len(self.result.posterior), dtype=float)

        # populate converted parameters
        self.result.posterior = self.priors.conversion_function(self.result.posterior)

        # Loop through each sample in the posterior
        for i in range(len(self.result.posterior[self.parameter_names[0]])):
            # Calculate Chieff, Chip, Vkick, Mf and Chif using NR fit functions
            # Dynamically retrieve the parameters for each function call
            # .iloc: positional, so this stays correct if the posterior index is not a
            # plain 0..N-1 RangeIndex
            params = {name: self.result.posterior[name].iloc[i] for name in self.parameter_names}
            self.chieff_p_data[i] = self.Chieff(**params)
            self.chip_p_data[i] = self.chip(**params)
            self.vkick_p_data[i] = self.Vkick_NRfit(**params)
            self.Mf_p_data[i] = self.Mfin_NRfit(**params)
            self.Chif_p_data[i] = self.Chifin_NRfit(**params)

        # update bilby result file
        self.result.posterior['chi_eff'] = self.chieff_p_data
        self.result.posterior['chi_p'] = self.chip_p_data
        self.result.posterior['final_mass_source'] = self.Mf_p_data
        self.result.posterior['final_spin'] = self.Chif_p_data
        self.result.posterior['final_kick'] = self.vkick_p_data
        # bilby's `save` sampler kwarg may be a bool or a format name; save_to_file only
        # accepts a format name, so a bare True would produce a bogus extension.
        save_format = self.sampler_kwargs.get('save')
        self.result.save_to_file(
            overwrite = True,
            extension = save_format if isinstance(save_format, str) else 'json',
        )

        print('Finished Chieff/Chip/Vkick/Mfinal/Chifinal calculation')

        print('End')
        return self


    def run_analysis_preserving(self, random_samples, num_cpus = 10):
        """
        Run the genealogical reconstruction analysis that preserves the original posteriors of the child BHs

        Parameters:
        ----------
        - random_samples (array): Randomly generated samples of different ancestral masses (with a total mass normalized to unity) and spin parameters.
        - num_cpus (int, optional): Number of CPUs for parallel execution (default: 10)
        """

        print('Start')

        # Evaluate final mass (i.e., the fraction of the total mass that remains in the final black hole) and spin for each sample
        print('Started Mfinal/Chifinal/Vkick calculation')
        mf_list = batch_parallel_evaluate(self.Mfin_NRfit, num_cpus, **random_samples)
        af_list = batch_parallel_evaluate(self.Chifin_NRfit, num_cpus, **random_samples)
        vf_list = batch_parallel_evaluate(self.Vkick_NRfit, num_cpus, **random_samples)
        print('Finished Mfinal/Chifinal/Vkick calculation')
        
        # Loop over posterior samples
        idxs = []
        for a_target in tqdm.tqdm(self.posteriors_candidate['spin'], desc="Processing posteriors"):
            # Compute spin difference
            diff_spin_all = np.nan_to_num(np.abs(a_target - af_list), nan=np.inf)
            idx = np.argmin(diff_spin_all) # Index of best match
            idxs.append(idx)

        posterior = {k: params[idxs] for k, params in random_samples.items()}
        posterior['final_mass_source'] = self.posteriors_candidate['mass_source']
        posterior['final_spin_original'] = self.posteriors_candidate['spin']
        posterior['final_spin'] = af_list[idxs]
        posterior['final_kick'] = vf_list[idxs]
        # Rescales the value of individual masses to match the inferred total mass
        total_mass_source = posterior['final_mass_source'] / mf_list[idxs]
        posterior['mass_1_source'] *= total_mass_source
        posterior['mass_2_source'] *= total_mass_source
        names = self.Chieff.__code__.co_varnames[:self.Chieff.__code__.co_argcount]
        posterior['chi_eff'] = self.Chieff(**{k: posterior[k] for k in names if k in posterior})
        posterior['chi_p'] = self.chip(**{k: posterior[k] for k in names if k in posterior})

        # Print how many samples exceeded tolerance
        for tolerance in 1e-2, 1e-3:
            diff = np.abs(posterior['final_spin'] - posterior['final_spin_original'])
            count = np.sum(diff > tolerance)
            print(f"Number of samples with tolerance > {tolerance}: {count} out of {len(idxs)}")

        self.result = dict(posterior = posterior)
        posterior = {k: list(posterior[k]) for k in posterior}
        save_path = os.path.join(self.outdir, f"{self.label}_result.json")
        with open(save_path, 'w') as f:
            json.dump(dict(posterior = posterior), f)
        print(f"Results saved to {save_path}")

        print('End')
        return self


def batch_parallel_evaluate(NRfit_func, num_cpus = 1, **params_dict):
    """
    Evaluate a function in parallel over parameter arrays.

    Parameters:
    ----------
    - function (func): callable
    - **params_dict (dict): Each key is a parameter name and each value is a NumPy array of samples.

    Returns:
    - np.ndarray: Results from evaluating function on input samples.
    """

    # Extract the names of all required parameters/argument names from NRfit_func’s signature
    required_params = list(inspect.signature(NRfit_func).parameters.keys())

    # Check for missing params -- Ensure all required parameters are there
    missing = [p for p in required_params if p not in params_dict]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    # Ensure all arrays are the same length
    length_set = {len(params_dict[p]) for p in required_params}
    if len(length_set) > 1:
        raise ValueError("All parameter arrays must be the same length.")

    # Order the inputs correctly and zip for parallel evaluation
    inputs = list(zip(*[params_dict[param] for param in required_params]))

    # Parallel evaluation
    results = p_map(lambda args: as_float(NRfit_func(*args)), inputs, num_cpus=num_cpus)
    return np.array(results)
