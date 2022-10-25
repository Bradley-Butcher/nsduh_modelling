import numpyro
import pandas as pd

from numpyro.distributions import Geometric, Normal, Uniform, Categorical, Dirichlet, Bernoulli
import jax.numpy as jnp

from nsduh_model.load import load_nsduh

def model(crime, race):
    # crime prior per race
    crime_prior = numpyro.sample("crime_prior", Uniform(0, 1).expand([3]))
    arrest_prior = numpyro.sample("arrest_prior", Uniform(0, 1).expand([3]))

    with numpyro.plate("data", len(data)) as i:
        race = numpyro.sample("race", Dirichlet(3), obs=race)

        # number of crimes - geometric distribution
        n_crimes = numpyro.sample(
            "n_crimes",
            numpyro.distributions.Geometric(crime_prior[race]),
            obs=crime
        )

        alpha = numpyro.sample("alpha", Uniform(0, 1).expand([n_crimes]))
        arrest_prob = arrest_prior[race] + alpha * n_crimes

        arrest = numpyro.sample(
            "arrest",
            numpyro.distributions.Bernoulli(arrest_prob,
            obs=arrest
        )

def learn(crime_col):
    nuts_kernel = NUTS(crimes)
    mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=100)
    race = data[race].astype('category')
    race = race.apply(lambda x: race.cat.codes)
    crime = data[crime_col]
    mcmc.run(rng_key, crime=crime, race=race)

if __name__ == "__main__":
    data = load_nsduh(nrows=1000) 
    learn('Total Drug Use')

    

