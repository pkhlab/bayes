# bayes

### This is a framework developed to fit systems models. The framework implemented Bayesian-inference fitting as well as a hierarchical mixture architecture. With these features, the framework could be used to fit to biological data, modeling not only proposed hypothesis, but also heterogeneity as often encountered in real-life data.



## Usage

python run_pymc.py --data expt_data.csv --model model1.py --param params_expt.yaml

expt_data.csv: The data you would like to fit to
model1.py: a custom PyMC-based script where you could define your theoretical model to fit to the data
params_expt.yaml: a YAML file where you could define processing options of the input data, priors of the model parameters, and settings of the MCMC sampling of the inference







