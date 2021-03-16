# import stan
# if __name__=='__main__':
# 	# https://pystan.readthedocs.io/en/latest/getting_started.html
# 	schools_code = """
# 	data {
# 	  int<lower=0> J;         // number of schools
# 	  real y[J];              // estimated treatment effects
# 	  real<lower=0> sigma[J]; // standard error of effect estimates
# 	}
# 	parameters {
# 	  real mu;                // population treatment effect
# 	  real<lower=0> tau;      // standard deviation in treatment effects
# 	  vector[J] eta;          // unscaled deviation from mu by school
# 	}
# 	transformed parameters {
# 	  vector[J] theta = mu + tau * eta;        // school treatment effects
# 	}
# 	model {
# 	  target += normal_lpdf(eta | 0, 1);       // prior log-density
# 	  target += normal_lpdf(y | theta, sigma); // log-likelihood
# 	}
# 	"""
# 	schools_data = {"J": 8,
#                 "y": [28,  8, -3,  7, -1,  1, 18, 12],
#                 "sigma": [15, 10, 16, 11,  9, 11, 10, 18]}
# 	posterior = stan.build(schools_code, data=schools_data, random_seed=1)
# 	fit = posterior.sample(num_chains=4, num_samples=1000)

# https://towardsdatascience.com/an-introduction-to-bayesian-inference-in-pystan-c27078e58d53
import pystan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
if __name__=='__main__':
	sns.set()  # Nice plot aesthetic
	np.random.seed(101)

	model = """
	data {
	    int<lower=0> N;
	    vector[N] x;
	    vector[N] y;
	}
	parameters {
	    real alpha;
	    real beta;
	    real<lower=0> sigma;
	}
	model {
	    y ~ normal(alpha + beta * x, sigma);
	}
	"""
	# Parameters to be inferred
	alpha = 4.0
	beta = 0.5
	sigma = 1.0

	# Generate and plot data
	x = 10 * np.random.rand(100)
	y = alpha + beta * x
	y = np.random.normal(y, scale=sigma)
	# Put our data in a dictionary
	data = {'N': len(x), 'x': x, 'y': y}

	# Compile the model
	sm = pystan.StanModel(model_code=model)

	# Train the model and generate samples
	fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)