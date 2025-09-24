#######################################
#                                     #
#            `potest.py`              #
#                                     #
#   Written by Manuel A. Buen-Abad    #
#           2020?, Sep. 2025          #
#                                     #
#######################################

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

import sys
import getopt

import pandas as pd
import numpy as np

import scipy.stats as sct
import scipy.integrate as sci

from tabulate import tabulate


# ****************************************

# ---------------------------------------------
#            C O N S T A N T S 
# ---------------------------------------------


_method_ = 'quad' # default integration method
_Nsamples_ = (2**16) # default number of samples for QMC integration method


# ---------------------------------------------
#          H E L P E R   F U N C T I O N
# ---------------------------------------------


def round_sig(x, sig=3):
    """
    Round a scalar or array to a specified number of significant digits.

    Parameters
    ----------
    x : float or array-like
        Input value(s) to round. Can be a scalar or a numpy array.
    sig : int, optional
        Number of significant digits to round to (default is 3).

    Returns
    -------
    float or numpy.ndarray
        Rounded value(s) with the specified number of significant digits.
    """

    x = np.asarray(x)
    
    # handle 0 separately (since log10(0) is undefined)
    mask = x != 0

    out = np.zeros_like(x, dtype=float)

    # order of magnitude: floor(log10(abs(x)))
    mags = np.floor(np.log10(np.abs(x[mask])))

    # scale so we can round to correct sig digits
    out[mask] = np.round(x[mask] / 10**mags, sig-1) * 10**mags

    return out


# ---------------------------------------------
#  C O M M A N D   L I N E   I N T E R F A C E
# ---------------------------------------------


def parse_cli(argv):
    """
    Parse command-line arguments for the vendor-best probability estimator.

    Supports both short and long options:
        -h, --help           Show this message
        -m, --method         Integration method: 'mc' or 'quad' (default: 'quad')
        -s, --samples        Number of samples for Monte Carlo (default: 2**16)
        -p, --print_error    Boolean flag to print standard errors
        -e, --example        Compute built-in example dataset

    Parameters
    ----------
    argv : list of str
        Command-line arguments, typically sys.argv[1:].

    Returns
    -------
    tuple
        method : str
            Integration method ('mc' or 'quad').
        n_samples : int
            Number of samples for Monte Carlo integration (rounded to nearest power of two).
        print_error : bool
            Whether to print standard errors.
        compute_example : bool
            Whether to compute the built-in example dataset.
    """

    method = _method_
    n_samples = _Nsamples_
    print_error = False
    compute_example = False

    help_msg = (
        f"Usage: python {sys.argv[0]} "
        "[-h] [-m <mc|quad>] [-s <int>] [-p] [-e]\n\n"
        "Options:\n"
        "  -h, --help           Show this message\n"
        "  -m, --method=        Integration method: mc or quad (default: quad)\n"
        "  -s, --samples=       Number of samples (default: 2**16)\n"
        "  -p, --print_error    Print error estimates\n"
        "  -e, --example        Compute example (from README file)\n"
    )

    try:
        # short: h (help), m: (method), s: (samples), p (print_error)
        opts, _ = getopt.getopt(argv,
                                "hm:s:pe",
                                ["help",
                                 "method=",
                                 "samples=",
                                 "print_error",
                                 "example"])
    except getopt.GetoptError:
        raise SystemExit(help_msg)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            raise SystemExit(help_msg)
        elif opt in ("-m", "--method"):
            if arg not in ("mc", "quad"):
                raise SystemExit("Invalid --method value. Use 'mc' or 'quad'.")
            method = arg
        elif opt in ("-s", "--samples"):
            try:
                n_samples = int(2**np.round(np.log2(arg)))
                if n_samples <= 0:
                    raise ValueError
            except ValueError:
                raise SystemExit("--samples must be a positive integer.")
        elif opt in ("-p", "--print_error"):
            print_error = True
        elif opt in ("-e", "--example"):
            compute_example = True

    return method, n_samples, print_error, compute_example


# ---------------------------------------------
#     D I S T R I B U T I O N S
# ---------------------------------------------


class BetaLike:
    """
    Beta-like distribution wrapper compatible with scipy.stats.beta interface.

    Converts a vendor rating and review count into Beta distribution parameters.

    Parameters
    ----------
    rating : float
        Observed rating on the provided scale.
    nreviews : float
        Number of reviews used to compute the distribution parameters.
    scale : float, optional
        Maximum rating scale (default 1.0 for normalized ratings).

    Methods
    -------
    pdf(x)
        Probability density function evaluated at x.
    cdf(x)
        Cumulative distribution function evaluated at x.
    sf(x)
        Survival function S(x) = 1 - F(x) evaluated at x.
    ppf(q)
        Percent point function (inverse CDF) at quantile(s) q.
    rvs(size=None, random_state=None)
        Draw random variates from the Beta-like distribution.
    __call__(x)
        Alias for pdf(x) for backward compatibility.
    """

    def __init__(self, rating: float, nreviews: float, scale: float = 1.0):
        """
        Initialize the BetaLike distribution with alpha and beta parameters.

        Parameters
        ----------
        rating : float
            Observed rating on the given scale.
        nreviews : float
            Number of reviews.
        scale : float, optional
            Maximum rating scale (default 1.0).
        """

        p = float(rating) / float(scale)
        self.alpha = p * float(nreviews) + 1.0
        self.beta = (1.0 - p) * float(nreviews) + 1.0
        # reference to scipy.stats.beta used for method implementations
        self._dist = sct.beta

    def pdf(self, x):
        """
        Compute the probability density function at a given value or array of values.

        Parameters
        ----------
        x : float or array-like
            Value(s) at which to evaluate the PDF.

        Returns
        -------
        float or numpy.ndarray
            PDF evaluated at x.
        """

        return self._dist.pdf(x, a=self.alpha, b=self.beta)

    def cdf(self, x):
        """
        Compute the cumulative distribution function at a given value or array of values.

        Parameters
        ----------
        x : float or array-like
            Value(s) at which to evaluate the CDF.

        Returns
        -------
        float or numpy.ndarray
            CDF evaluated at x.
        """

        return self._dist.cdf(x, a=self.alpha, b=self.beta)

    def sf(self, x):
        """
        Compute the survival function S(x) = 1 - F(x) at a given value or array of values.

        Parameters
        ----------
        x : float or array-like
            Value(s) at which to evaluate the survival function.

        Returns
        -------
        float or numpy.ndarray
            Survival function evaluated at x.
        """

        return self._dist.sf(x, a=self.alpha, b=self.beta)

    def ppf(self, q):
        """
        Compute the percent point function (inverse CDF) at a given quantile or array of quantiles.

        Parameters
        ----------
        q : float or array-like
            Quantile(s) in the open interval (0, 1).

        Returns
        -------
        float or numpy.ndarray
            Value(s) corresponding to the given quantiles.
        """

        return self._dist.ppf(q, a=self.alpha, b=self.beta)

    def rvs(self, size=None, random_state=None):
        """
        Draw random variates from the Beta-like distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Number of random samples to draw.
        random_state : int, RandomState, or None, optional
            Random seed or generator for reproducibility.

        Returns
        -------
        float or numpy.ndarray
            Random variate(s) from the distribution.
        """

        return self._dist.rvs(a=self.alpha, b=self.beta, size=size, random_state=random_state)

    def __call__(self, x):
        """
        Allow the instance to be called like a function (alias for pdf).

        Parameters
        ----------
        x : float or array-like
            Value(s) at which to evaluate the PDF.

        Returns
        -------
        float or numpy.ndarray
            PDF evaluated at x.
        """

        return self.pdf(x)

    def __repr__(self):
        """
        Return an informal string representation for debugging purposes.

        Returns
        -------
        str
            Informal representation including alpha and beta parameters.
        """

        return f"BetaLike(alpha={self.alpha:.4g}, beta={self.beta:.4g})"


# ---------------------------------------------
#     C O M P U T A T I O N S
# ---------------------------------------------


def superior_experience_prob(pdf_ppf_candidate: tuple,
                             cdf_list: list,
                             method: str = _method_,
                             n_samples: int = _Nsamples_,
                             seed: int = 0):
    """
    Estimate the probability that a candidate vendor provides the best outcome.

    Uses either quasi-Monte Carlo with Sobol sequences or quadrature integration.

    Parameters
    ----------
    pdf_ppf_candidate : tuple
        Tuple of (pdf, ppf) callables for the candidate vendor.
    cdf_list : list of callables
        Competitor CDF functions.
    method : str, optional
        Integration method: 'mc' for Monte Carlo or 'quad' for quadrature (default: 'quad').
    n_samples : int, optional
        Number of QMC or MC samples (default: 2**16).
    seed : int, optional
        Random seed for pseudo-random fallback (default 0).

    Returns
    -------
    tuple
        (estimate, stderr) where estimate is the probability that the candidate
        is best, and stderr is the Monte Carlo standard error (always returned).
    """

    # Number of other vendors
    n_other = len(cdf_list)
    
    # If there are no competitors the candidate is trivially best
    if n_other == 0:
        return (1.0, 0.0)

    if method == 'quad':
        # For quadrature integration we need the vendor's PDF:
        vendor_pdf, _ = pdf_ppf_candidate

        # The integrand is the vendor's PDF times the product of the competitors' CDF:
        def integrand(x):
            
            # Stack competitor CDFs + vendor PDF along axis 0
            vals = np.vstack([vendor_pdf(x)] + [cdf_j(x) for cdf_j in cdf_list])

            # Multiply across rows
            res = np.prod(vals, axis=0)

            # Flatten out result
            if len(res) == 1:
                res = float(res[0])

            return res

        est, stderr = sci.quad(integrand,
                           a=0,
                           b=1,
                           epsabs=1.e-6,
                           epsrel=1.e-3,
                           limit=1_000)

    elif method == 'mc':
        # For (Q)MC integration we need the vendor's PPF
        _, vendor_ppf = pdf_ppf_candidate

        # Prepare fallback RNG for pseudo-random draws
        rng = np.random.default_rng(seed)

        # Prefer scipy.stats.qmc.Sobol for QMC draws. Fall back to RNG on failure.
        try:
            sob = sct.qmc.Sobol(d=1, scramble=True)
            u = sob.random(n_samples)
        except Exception:
            u = rng.random(size=n_samples)
        
        # Reshape draws
        u = u.reshape(-1) if (u.ndim == 2 and u.shape[1] == 1) else u

        # Clip values away from exact 0 and 1 to avoid ppf extremes
        low = np.nextafter(0.0, 1.0)
        high = np.nextafter(1.0, 0.0)
        u = np.clip(u, low, high)

        # Transform uniforms to Beta samples using vendor's PPF
        Finv_i = vendor_ppf(u)

        # Compute samples using product of competitors' CDFs
        samples = np.ones_like(u)
        for cdf_j in cdf_list:
            samples *= cdf_j(Finv_i)

        # Monte Carlo estimate and standard error
        est = float(np.mean(samples))
        var = float(np.var(samples, ddof=1)) if n_samples > 1 else 0.0
        stderr = np.sqrt(var / n_samples)

    else:
        raise ValueError(f"You passed --method {method}. It must be either 'mc' or 'quad'.")

    return (est, stderr)


# ---------------------------------------------
#     M A I N
# ---------------------------------------------


def main():
    """
    Interactive main routine for collecting vendor data and displaying rankings.

    - Collects vendor name, rating, and number of reviews.
    - Constructs a DataFrame with PDF, CDF, SF, and PPF callables.
    - Computes the probability of best experience for each vendor.
    - Sorts and prints vendors by estimated probability.

    Returns
    -------
    None
    """

    # Welcome
    print("\n")
    print("*" * 100)
    print("Welcome to `potest.py`, a python script to choose the best vendor for your needs!\nLet us help you make your best decision.")
    print("*" * 100)

    # Reading the arguments
    method, n_samples, print_err, compute_example = parse_cli(sys.argv[1:])

    if compute_example:
        # Use example data:
        
        # 4 vendors
        n = 4
        
        # 5-star scale
        scale = 5
        
        # Names
        names = ['A', 'B', 'C', 'D']
        
        # Ratings
        ratings = [3., 3.4, 4.2, 2.2]
        
        # Number of reviews
        nrevs = [65, 15, 55, 364]

        print(f"You are running our example data, which has {n} vendors with a {scale}-star rating scale:\n")
        print("...\t(name, rating, reviews) =", list(zip(names, ratings, nrevs)),"\n")
        
        rows = []
        for i in range(n):

            # Vendor dictionary
            vendor_dict = {}
            
            # Name
            vendor_dict["Vendor"] = names[i]

            # Rating
            vendor_dict["Rating"] = ratings[i]

            # Number of Reviews
            vendor_dict["No. of Reviews"] = nrevs[i]

            # Vendor distribution
            vendor_dist = BetaLike(rating=ratings[i], nreviews=nrevs[i], scale=scale)

            # Bind methods to keep DataFrame-compatible callables
            vendor_dict["PDF"] = vendor_dist.pdf
            vendor_dict["CDF"] = vendor_dist.cdf
            vendor_dict["PPF"] = vendor_dist.ppf
            vendor_dict["SF"] = vendor_dist.sf

            # Placeholders for results
            vendor_dict["Prob. of Best Exper."] = 0.0
            if print_err:
                vendor_dict["Std. Error Prob."] = 0.0
            
            # Append vendor data to rows
            rows.append(vendor_dict)

    else:
        # Ask user for vendor data

        # Ask for number of vendors
        n = int(input("Enter number of vendors: "))

        # Ask for rating scale
        scale = int(input("Enter rating scale (e.g. 5 for 5 stars, or 10 for 10 points): "))

        print("-" * 50)

        rows = []
        for i in range(n):
            vendor_dict = {}

            print(f"Please input the following data for vendor #{i+1}:")

            # Name
            name = input("...\tname: ").strip()
            vendor_dict["Vendor"] = name

            # Rating
            rating = float(input(f"...\trating (out of {scale}): ").strip())
            vendor_dict["Rating"] = rating

            # Number of Reviews
            nrev = float(input("...\tnumber of reviews: ").strip())
            vendor_dict["No. of Reviews"] = nrev

            if i < n-1:
                print("." * 50)

            # Vendor distribution
            vendor_dist = BetaLike(rating=rating, nreviews=nrev, scale=scale)

            # Bind methods to keep DataFrame-compatible callables
            vendor_dict["PDF"] = vendor_dist.pdf
            vendor_dict["CDF"] = vendor_dist.cdf
            vendor_dict["PPF"] = vendor_dist.ppf
            vendor_dict["SF"] = vendor_dist.sf

            # Placeholders for results
            vendor_dict["Prob. of Best Exper."] = 0.0
            if print_err:
                vendor_dict["Std. Error Prob."] = 0.0

            # Append vendor data to rows
            rows.append(vendor_dict)

    # Turning rows into DataFrame
    data = pd.DataFrame(rows)

    print("*" * 75)
    print("Computing probabilities of best experience:")

    # Looping over all vendors
    for vendor in data["Vendor"]:

        # Mask for vendor row
        vendor_mask = (data["Vendor"] == vendor)

        # Mask for competitors
        competitors_mask = ~vendor_mask

        # Vendor PDF and PPF
        pdf_cand = data[vendor_mask]["PDF"].to_list()[0]
        ppf_cand = data[vendor_mask]["PPF"].to_list()[0]

        # Competitors' CDF
        cdf_list = data[competitors_mask]["CDF"].to_list()
        
        # Computing probability
        prob, stderror = superior_experience_prob(pdf_ppf_candidate=(pdf_cand, ppf_cand),
                                                  cdf_list=cdf_list,
                                                  method=method,
                                                  n_samples=n_samples)

        # Saving results to appropriate entry
        data.loc[vendor_mask, "Prob. of Best Exper."] = prob
        if print_err:
            data.loc[vendor_mask, "Std. Error Prob."] = stderror

    print("\n...\tDone!")

    # Sorting dataframe according to probabilities
    data.sort_values(by="Prob. of Best Exper.", ascending=False, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Columns to display for the user
    num_cols = ["Prob. of Best Exper."]
    if print_err:
        num_cols += ["Std. Error Prob."]
    display_cols = ["Vendor", "Rating", "No. of Reviews"] + num_cols
    
    # Dataframe to display for the user
    data_display = data[display_cols].copy()
    data_display[num_cols] = data_display[num_cols].map(lambda x: round_sig(x, 3))

    # Print result
    print("\nRanking:\n")
    print(tabulate(data_display, headers='keys', tablefmt='github'))
    print("\nBest vendor:", data.iloc[0, 0], "\n")


# ********************************************************************************
# ********************************************************************************
# ********************************************************************************


if __name__ == "__main__":
    main()