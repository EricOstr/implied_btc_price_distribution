# BTC Options Implied Price Distribution

This ongoing project is a dashboard displaying the implied future BTC price distribution inferred from options prices (see live dashboard [here](http://ec2-3-250-69-124.eu-west-1.compute.amazonaws.com:8888/)). The dashboard updates in 10 minute intervals with data collected from [Deribit BTC options exchange](https://docs.deribit.com/). 

The implied price distribution's densities are inferred by constructing butterflies at various strike prices and comparing the price of the butterfly to the the maximum payoff of the butterfly. Data points are smoothed using a gaussian filter before they are used to construct splines using [scipy.interpolate.CubicSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html). The splines are then normalized to create the final probability density function (PDF). From this PDF, we infer probabilities in 10 percentile increments.

The second part of the project consists of [Monte Carlo options pricing model](#sim) used to price a specific option chosen by the user



The application is deployed as a Docker container using AWS. 


#### Improvement Ideas:
* Create a model forecasting the probability distribution of BTC price for given date - compare performance of this model to option market's prediction
* Calculate option price with monte carlo using forecasted price distribution

#### Improvement Ideas (Technical):
* Migrate to AWS CDK for CI/CD
* Separate from monolithic architecture to a micro service based one (dedicated service for MC simulation, volatility modelling etc)
* Use C++ for faster MC simulation



## BTC Options Implied Price Distribution
![price_dist](https://user-images.githubusercontent.com/45294679/185405696-2a775e05-fdab-4154-929e-310610f8305a.png)



## Monte Carlo Options Pricing Model
![sim](https://user-images.githubusercontent.com/45294679/185405847-dcaf7bcd-60fa-45b1-9b8f-1e99cfa15d85.png)
