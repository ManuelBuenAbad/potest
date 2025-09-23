"""Vendor-best-probability estimator using Sobol quasi-Monte Carlo and a Beta-like wrapper.

This module implements an interactive CLI and a DataFrame layout that stores
callables for each vendor: PDF, CDF, SF and PPF. It estimates the probability
that a given vendor produces the best outcome using Sobol quasi-Monte Carlo
when available and falls back to pseudo-random uniforms otherwise.

Only minimal changes were made to preserve the original control flow. The
implementation prefers ``scipy.stats.qmc.Sobol`` and the Monte Carlo routine
always returns a tuple ``(estimate, stderr)`` so callers receive a consistent
return value.
"""

import pandas as pd
import numpy as np

from scipy.stats import qmc
import scipy.stats as st

from tabulate import tabulate

# ****************************************
# ----------------------------------------
#     D I S T R I B U T I O N S
# ----------------------------------------


class BetaLike:
    """Small wrapper exposing the same interface as ``scipy.stats.beta``.

    Parameters
    ----------
    rating : float
        Observed rating on the provided ``scale`` (e.g. 4.7 for a 5-point scale).
    nreviews : float
        Number of reviews used to compute distribution parameters.
    scale : float, optional
        Rating scale maximum (default 1.0 for already-normalized ratings).

    Notes
    -----
    The mapping from rating and review count to Beta parameters is:

        p = rating / scale
        alpha = p * nreviews + 1
        beta = (1 - p) * nreviews + 1

    The wrapper exposes ``pdf``, ``cdf``, ``sf``, ``ppf`` and ``rvs`` methods
    and implements ``__call__`` as an alias to ``pdf`` for backward
    compatibility with code expecting a callable returning a density.
    """

    def __init__(self, rating: float, nreviews: float, scale: float = 1.0):
        """Create a BetaLike instance and compute alpha/beta parameters."""
        p = float(rating) / float(scale)
        self.alpha = p * float(nreviews) + 1.0
        self.beta = (1.0 - p) * float(nreviews) + 1.0
        # reference to scipy.stats.beta used for method implementations
        self._dist = st.beta

    def pdf(self, x):
        """Return the probability density function evaluated at ``x``.

        Accepts scalar or array-like ``x`` and returns an array-like result.
        """
        return self._dist.pdf(x, a=self.alpha, b=self.beta)

    def cdf(self, x):
        """Return the cumulative distribution function evaluated at ``x``."""
        return self._dist.cdf(x, a=self.alpha, b=self.beta)

    def sf(self, x):
        """Return the survival function ``S(x) = 1 - F(x)`` evaluated at ``x``."""
        return self._dist.sf(x, a=self.alpha, b=self.beta)

    def ppf(self, q):
        """Return the percent point function (inverse CDF) at quantile(s) ``q``.

        ``q`` may be scalar or array-like in the open unit interval. Callers
        should avoid exact 0 or 1 values for numerical stability.
        """
        return self._dist.ppf(q, a=self.alpha, b=self.beta)

    def rvs(self, size=None, random_state=None):
        """Draw random variates. Parameters forwarded to ``scipy.stats.beta.rvs``."""
        return self._dist.rvs(a=self.alpha, b=self.beta, size=size, random_state=random_state)

    def __call__(self, x):
        """Callable alias for ``pdf`` (backward compatibility)."""
        return self.pdf(x)

    def __repr__(self):
        """Informal representation for debugging."""
        return f"BetaLike(alpha={self.alpha:.4g}, beta={self.beta:.4g})"


# ----------------------------------------
#     C O M P U T A T I O N S
# ----------------------------------------


def superior_experience_prob(ppf_list: list,
                             sdf_candidate,
                             n_samples: int = 2 ** 16,
                             seed: int = None):
    """Estimate the probability the candidate vendor yields the superior outcome.

    The routine uses Sobol quasi-Monte Carlo when available. Uniform draws on
    ``[0,1)`` are transformed through each competitor's ``ppf`` (inverse CDF)
    to obtain Beta samples. For each draw the maximum across competitors is
    computed and the candidate's survival function is evaluated at that
    maximum. The mean of those survival values equals the desired probability.

    Parameters
    ----------
    ppf_list : list
        Sequence of callables. Each callable maps a value in (0,1) to a
        realization for a competitor vendor (these are the competitor PPFs).
    sdf_candidate : callable
        Callable mapping scalar or array-like ``t`` to ``S_candidate(t)``.
    n_samples : int, optional
        Number of quasi-Monte Carlo draws (default 2**16).
    seed : int or None, optional
        Seed used to initialize the fallback pseudo-random number generator.

    Returns
    -------
    tuple
        ``(estimate, stderr)`` where ``stderr`` is the Monte Carlo standard
        error computed from sample variance. Always returned.
    """

    n_other = len(ppf_list)

    # If there are no competitors the candidate is trivially best
    if n_other == 0:
        return (1.0, 0.0)

    # Prepare fallback RNG for pseudo-random draws
    rng = np.random.default_rng(seed)

    # Prefer scipy.stats.qmc.Sobol for QMC draws. Fall back to RNG on failure.
    try:
        sob = qmc.Sobol(d=n_other, scramble=True)
        u = sob.random(n_samples)
    except Exception:
        u = rng.random(size=(n_samples, n_other))

    # Clip values away from exact 0 and 1 to avoid ppf extremes
    low = np.nextafter(0.0, 1.0)
    high = np.nextafter(1.0, 0.0)
    u = np.clip(u, low, high)

    # Transform uniforms to Beta samples using each competitor's PPF
    samples = np.empty_like(u)
    for j, ppf in enumerate(ppf_list):
        samples[:, j] = ppf(u[:, j])

    # Maximum across competitors per draw
    maxima = np.max(samples, axis=1)

    # Evaluate candidate survival at those maxima
    svals = np.asarray(sdf_candidate(maxima))

    # Estimate and Monte Carlo standard error
    est = float(np.mean(svals))
    var = float(np.var(svals, ddof=1)) if n_samples > 1 else 0.0
    stderr = np.sqrt(var / n_samples)

    return (est, stderr)


# ----------------------------------------
#     M A I N
# ----------------------------------------


def main():
    """Interactive main that collects vendor data and prints rankings.

    The function reads the number of vendors and rating scale, then iteratively
    captures vendor name, rating and number of reviews. It constructs a
    DataFrame with callables for PDF, CDF, SF and PPF and computes the
    probability each vendor yields the best outcome using QMC.
    """
    # Ask for number of vendors
    n = int(input("Enter number of vendors: "))

    # Ask for rating scale
    scale = int(input("Enter rating scale (e.g. 5 for 5 stars, or 10 for 10 points): "))

    print("*" * 75)

    rows = []
    for i in range(n):
        vendor_dict = {}

        print(f"Please input the following data for vendor #{i+1}:")

        # Name
        name = input("...	name: ").strip()
        vendor_dict["Vendor"] = name

        # Rating
        rating = float(input(f"...	rating (out of {scale}): ").strip())
        vendor_dict["Rating"] = rating

        # Number of Reviews
        nrev = float(input("...	number of reviews: ").strip())
        vendor_dict["No. of Reviews"] = nrev

        # Vendor distribution
        vendor_dist = BetaLike(rating=rating, nreviews=nrev, scale=scale)

        # Bind methods to keep DataFrame-compatible callables
        vendor_dict["PDF"] = vendor_dist.pdf
        vendor_dict["CDF"] = vendor_dist.cdf
        vendor_dict["SF"] = vendor_dist.sf
        vendor_dict["PPF"] = vendor_dist.ppf

        # Placeholders for results
        vendor_dict["Prob. of Best Exper."] = 0.0
        vendor_dict["Std. Error Prob."] = 0.0

        rows.append(vendor_dict)

        print("." * 50)

    # Turning rows into DataFrame
    data = pd.DataFrame(rows)

    print("*" * 75)
    print("Computing probabilities of best experience:")

    for vendor in data["Vendor"]:
        vendor_mask = (data["Vendor"] == vendor)
        others_mask = (data["Vendor"] != vendor)

        ppf_list = data[others_mask]["PPF"].to_list()
        sdf_candidate = data[vendor_mask]["SF"].to_list()[0]

        prob, stderror = superior_experience_prob(ppf_list=ppf_list,
                                                  sdf_candidate=sdf_candidate)

        data.loc[vendor_mask, "Prob. of Best Exper."] = prob
        data.loc[vendor_mask, "Std. Error Prob."] = stderror

    print("...	Done!")

    # Sorting according to probabilities
    data.sort_values(by="Prob. of Best Exper.", ascending=False, inplace=True)
    data.reset_index(drop=True, inplace=True)

    display_cols = ["Vendor", "Rating", "No. of Reviews", "Prob. of Best Exper.", "Std. Error Prob."]

    # Print result
    print("Ranking:")
    print(tabulate(data[display_cols], headers='keys', tablefmt='github'))
    print("Best vendor:", data.iloc[0, 0])


# ********************************************************************************
# ********************************************************************************
# ********************************************************************************


if __name__ == "__main__":
    main()