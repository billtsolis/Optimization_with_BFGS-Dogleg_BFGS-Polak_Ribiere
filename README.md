## Introduction

An investor wants to invest in investment products whose returns are related to 4 national commercial sectors of the USA:

(K1) Sales of computers-electronics.

(K2) Sales of defense equipment.

(K3) Sales of motor vehicles and parts.

(K4) Sales of metals.

These specific investment products offer monthly returns equal to the percentage change in the served orders (in $ billion) overall in the USA, between the investment time (start month) and the withdrawal time (end month) from the investment product. For example, if $1 is invested in the computers-electronics sector in month t where the sector had sales of $17950 billion and the investor withdraws in month t+k where the sector recorded sales of $21658 billion, then the return is:

$21658/
$17950
≈ $1.20657

meaning the return on his investment is approximately 20.66% for the period of k months. Conversely, if he withdraws at some other time t + m where the sector recorded sales of $16108 billion, then the return is:

$16108/
$17950
≈ $0.89738

meaning the return on his investment in this case is approximately -10.26% for m months.

The provided data files named:

(A1) data ComputersElectronicProducts.csv

(A2) data DefenseCapitalGoods.csv

(A3) data MotorVehiclesParts.csv

(A4) data PrimaryMetals.csv

contain, respectively for each of the 4 commercial sectors, the monthly values (in $ billion) of served orders in the USA for the period between February 1992 and September 2023, i.e., 380 monthly values for each sector.

The aim of this study is to assist the investor in deciding what percentage of his capital he should invest in each of the 4 commercial sectors K1-K4 in order to maximize the expected relative return of his investment, while simultaneously minimizing the risk expressed as the correlation of the performance values of the 4 sectors. For this purpose, appropriate optimization will be performed using all available historical data. We will assume that the investment start time occurs immediately after the passage of the 380 available months.

## Detailed Description

The values provided in each of the files A1-A4 correspond to the monthly sales of the commercial sectors K1-K4 in $ billion. If February 1992 is considered as the initial time of the time horizon (the 1st recorded value in each file), then we can convert all given values into corresponding relative return values with respect to the base value of the 1st month as follows:

$$
R_{ij} = \frac{{\text{{sales of sector }} K_i \text{{ in month }} j}}{{\text{{sales of sector }} K_i \text{{ in month }} 1}} - 1,
$$

for each \( j = 1, 2, ..., 380 \), and \( i = 1, 2, 3, 4 \). Τhe received relative return values \( R_{ij} \) are provided in all 4 cases. Let's denote, respectively, with \( R(i) \) the vector of relative returns:

$$
R(i) = (R_{1j}, R_{2j}, ..., R_{380j})^T,
$$

and with \( \bar{R}(i) \) the mean of the relative returns:

$$
\bar{R}(i) = \text{{mean}}(R(i)) = \frac{1}{380} \sum_{j=1}^{380} R_{ij},
$$

For each \( i = 1, 2, 3, 4 \), let \( M \) be the covariance matrix of the relative return values:

$$
M = \begin{pmatrix} \text{cov}(R(1),R(1)) & \text{cov}(R(1),R(2)) & \text{cov}(R(1),R(3)) & \text{cov}(R(1),R(4)) \\
\text{cov}(R(2),R(1)) & \text{cov}(R(2),R(2)) & \text{cov}(R(2),R(3)) & \text{cov}(R(2),R(4)) \\
\text{cov}(R(3),R(1)) & \text{cov}(R(3),R(2)) & \text{cov}(R(3),R(3)) & \text{cov}(R(3),R(4)) \\
\text{cov}(R(4),R(1)) & \text{cov}(R(4),R(2)) & \text{cov}(R(4),R(3)) & \text{cov}(R(4),R(4)) \end{pmatrix},
$$

where \( cov(R(i),R(j)) \) represents the covariance between \( R(i) \) and \( R(j) \), for \( i, j = 1, 2, 3, 4 \).

If we denote by \( w_i \) belonging to the interval [0.0, 1.0],, the percentage of capital invested in the respective commercial sectors K1-K4, then the objective of the problem is to maximize the expected return while simultaneously minimizing the risk. Thus, we need to solve the maximization problem:

$$
\max_{w} f(w) = w^T \bar{R} - \lambda w^T M w,
$$

where \( w = (w(1), w(2), w(3), w(4)) \) are the participation rates per commercial sector, λ > 0 is a parameter determining the importance of the risk, and:

$$
\bar{R} = [\bar{R}(1), \bar{R}(2), \bar{R}(3), \bar{R}(4)],
$$

is the vector of mean returns. The above problem can be written in a more detailed form as:

$$
\min_{w} F(w) = - \sum_{i=1}^{4} w_i \bar{R}(i) - \lambda \sum_{j=1}^{4} \sum_{k=1}^{4} w_j w_k \text{cov}(R(j),R(k)),
$$

subject to the constraint:

$$
\sum_{i=1}^{4} w_i = 1.
$$

Since incorporating such constraints is currently not feasible, one can consider the following variables of the problem:

$$
x_1, x_2, x_3, x_4 \in [0.0, 1.0],
$$

from which the participation rates are derived as follows:

$$
w_i = \frac{x_i}{\sum_{j=1}^{4} x_j}, \quad i = 1, 2, 3, 4.
$$

Therefore, the objective function \( F(w) \) now becomes a function of \( x(1), x(2), x(3), x(4) \), which are the variables for optimization within the search space \( X \) being defined as the set [0.0, 1.0]^4. For the purposes of this work, let's consider λ = 1.5 in \( F(w) \).

## Optimization Algorithms

The optimization problem was solved using the following optimization algorithms:

1. **BFGS with linear search with Wolfe conditions**
2. **Dogleg BFGS**
3. **Polak-Ribiere conjugate gradient method**
