# bayes

This is a framework developed to fit systems models. The framework implemented Bayesian-inference fitting as well as a hierarchical mixture architecture. With these features, the framework could be used to fit to biological data, modeling not only proposed hypothesis, but also heterogeneity as often encountered in real-life data.

## Dependencies
- Python 3.10
- pymc >= 5.0
- sunode >= 0.4
- nutpie >= 0.5
- arviz >= 0.15
- pytensor >= 2.8
- numpy >= 1.23
- scipy >= 1.10
- pandas >= 1.5

## Usage

python run_pymc.py --data expt_data.csv --model model1.py --param params_expt.yaml

**expt_data.csv:** observed data

**model1.py:** a custom PyMC-based script describing the theoretical model proposed to fit to the data

**params_expt.yaml:** a YAML file to define processing options of the input data, priors of the model parameters, and settings of the MCMC sampling of the inference


## Project Structure

```.
├── analysis_notebooks
│   ├── ca_uptake
│   └── cyto_ER_shuttling
├── code
│   ├── run_pymc.py: pymc script
│   └── sma.py
├── data
│   └── observed_data
├── LICENSE
├── model
│   ├── ca_uptake
│   └── cyto_ER_shuttling
├── README.md
```



** analysis_scripts:

** code

** data/observed_data:

** model:




### To cite:
Fang, X., Varughese, P., Osorio-Valencia, S., Zima, A.V., P.M. Kekenes-Huskey (2025). A Bayesian framework for systems model refinement and selection of calcium signaling

### Funding
Research reported in this publication was supported by the Maximizing Investigators’ Research Award (MIRA) (R35) from the National Institute of General Medical Sciences (NIGMS) of the National Institutes of Health (NIH) under grant number GM148284 to P.M.K.H. 
This work used Expanse at San Diego Supercomputer Center (SDSC) through allocation CHE140116 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.





