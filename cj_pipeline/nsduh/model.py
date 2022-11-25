# %%
import numpyro
import pandas as pd

from numpyro.distributions import Uniform, Bernoulli, Poisson, Normal
import jax.numpy as jnp
import jax

# numpyro import NUTS and MCMC
from numpyro.infer import NUTS, MCMC


from cj_pipeline.nsduh.load import load_nsduh

def cont_model(crime, arrests, race):
    # crime prior per race
    crime_prior = numpyro.sample("crime_prior", Uniform(0, 1).expand([3]))
    arrest_scale = numpyro.sample("arrest_scale", Uniform(0, 1).expand([3]))
    arrest_shift = numpyro.sample("arrest_shift", Uniform(0, 1).expand([3]))

    with numpyro.plate("data", len(data)) as i:
        # number of crimes - geometric distribution
        crime_commited = numpyro.sample(
            "crime_commited",
            Poisson(crime_prior[race]), 
            obs=crime
        )
        numpyro.sample(
            "arrest",
            Bernoulli(crime_commited * arrest_scale[race] + arrest_shift[race]),
            obs=arrests
        )


race_dict = {
    "White": 0,
    "Black": 1,
    "Hispanic": 2
}

arrest_dict = {
    "Yes": 1,
    "No": 0
}

def learn(crime_col) -> MCMC:
    rng_key = jax.random.PRNGKey(0)
    nuts_kernel = NUTS(cont_model)
    mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=1000)

    race = data["Race"].map(race_dict)
    race = jnp.array(race)

    crime = data[crime_col].astype('int')
    crime = jnp.array(crime)

    arrests = data["Arrested"].map(arrest_dict)
    arrests = jnp.array(arrests)

    mcmc.run(rng_key, crime=crime, arrests=arrests, race=race)
    return mcmc

# %%

data = load_nsduh()

# %%
mcmc_model = learn("Sold Drugs Past Year")

# %%

import arviz as az
from matplotlib import pyplot as plt

samples = mcmc_model.get_samples()


crime_samples = Poisson(mcmc_model.get_samples()["crime_prior"]).sample(jax.random.PRNGKey(0),)

az.plot_dist(crime_samples[:, 0], label="White", color="blue", fill_kwargs={"alpha": 0.2})
az.plot_dist(crime_samples[:, 1], label="Black", color="red", fill_kwargs={"alpha": 0.2})
az.plot_dist(crime_samples[:, 2],  label="Hispanic", color="green", fill_kwargs={"alpha": 0.2})

# %%
az.plot_dist(samples["arrest_shift"][:, 0] + samples["arrest_scale"][:, 0], label="White", color="blue", fill_kwargs={"alpha": 0.2},)
az.plot_dist(samples["arrest_shift"][:, 1] + samples["arrest_scale"][:, 1],label="Black", color="red", fill_kwargs={"alpha": 0.2})
az.plot_dist(samples["arrest_shift"][:, 2] + samples["arrest_scale"][:, 2],  label="Hispanic", color="green", fill_kwargs={"alpha": 0.2})

plt.title("P(Arrested = True | Drugs Sold in a Year, Race = r)")

# %%

az_obj = az.from_numpyro(mcmc_model)
# %%
