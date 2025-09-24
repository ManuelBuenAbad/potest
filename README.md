# potest

**"POsiTive Experience Statistical Test": a simple code that, based on input data about vendors and their ratings, recommends which is most likely to result in a positive experience to the client.
Written by Manuel A. Buen-Abad.**

📄 Description
-----------------------------------------

Based on vendors $i$ with ratings $r_i$ (in a rating system of scale $s$) based on $N_i$ reviews, this program computes the probability for each of these vendors to provide a customer experience _better_ than all the other vendors.
We do this by modeling the probability density for a given vendor $i$ to give a positivie experience with the beta function:

$f_i(x) \equiv \text{Beta}(x;\, \alpha_i, \beta_i) = \frac{\Gamma(\alpha_i + \beta_i)}{\Gamma(\alpha_i) \Gamma(\beta_i)} \, x^{\alpha_i - 1} (1 - x)^{\beta_i - 1}$,

where $\Gamma(x)$ is the gamma function, $\alpha_i \equiv \left( \frac{r_i}{s} \right) \cdot N_i + 1$ and $\beta_i \equiv \left( 1 - \frac{r_i}{s} \right) \cdot N_i + 1$.
Clearly $\alpha-1$ and $\beta-1$ are the average number of "successes" or "failures" for customer experience.

For example, consider 4 vendors rated with the 5-stars system.
Assume the data for these vendors is as follows:
- Vendor A: 3 stars & 65 reviews,
- Vendor B: 3.4 stars & 15 reviews,
- Vendor C: 4.2 stars & 55 reviews, and
- Vendor D: 2.2 stars & 364 reviews.

![Example of Vendor PDFs](./figures/vendor_pdfs.png)

The probability we are interested is then

$P(x_i > x_j \ \forall j \neq i) = \int\limits_0^1 ... \int\limits_0^1 \ \prod\limits_{j \neq i} \ \mathrm{d}x_j \ f_j(x_j) \cdot \int\limits_{\max(x_j)}^1 \ \mathrm{d}x_i f_i(x_i)$.

Some manipulations converts this into

$P(x_i > x_j \ \forall j \neq i) = \int\limits_0^1 ... \int\limits_0^1 \ \prod\limits_{j \neq i} \ \mathrm{d}x_j \ f_j(x_j) \cdot S_i(\max(x_j))$,

where $S(x) = 1 - F(x)$ is the survival function, and $F(x)$ is the cumulative density function.
In our code, we perform these integrals numerically with the help of Quasi-Monte Carlo methods.

After running, our code yields

| Vendor | Rating | No. of Reviews | Prob. of Best Exper. | Std. Error Prob. |
| -----: | -----: | -------------: | -------------------: | ---------------: |
|      C |    4.2 |             55 |             0.921133 |      0.000685693 |
|      B |    3.4 |             15 |             0.077599 |      0.000322995 |
|      A |      3 |             65 |           0.00127018 |       3.7999e-05 |
|      D |    2.2 |            364 |          6.09201e-15 |       4.3544e-15 |

The third column is the probability computed by the integrals above.
Clearly C is the best vendor.


📋 Instructions
-----------------------------------------

In your terminal run:

`python potest.py`

and follow the instructions that appear in the command-line interface (CLI) to enter the following data:

- the number of vendors you are interested in,
- the rating scale (e.g. enter 5 for ratings based on the 5-star system, or 10 for 10-points),
- the name of the vendor,
- its rating (e.g. 4.5 if the vendor has 4.5 stars)
- its number of reviews.

After this, the code will compute the probability for each vendor to provide an overall experience to the client better than that of all the others.
Finally, the code will print out a table with that information, ranking each vendor from highest to lowest according to this probability.


❓ Requirements
-----------------------------------------

1. python (>= 3.7)
2. pandas (>= 1.0.0)
3. numpy (>= 1.17.0)
4. scipy (>= 1.7.0)