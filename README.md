# Optimization_with_BFGS-Dogleg_BFGS-Polak_Ribiere
BFGS with line search with Wolfe conditions, Dogleg BFGS, Polak Ribiere
# Introduction

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
