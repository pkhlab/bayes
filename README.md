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

```
.
├── analysis_scripts
│   ├── ca_uptake
│   │   ├── model_comparison_expt.ipynb: model comparison notebook for inference runs done on experimental data
│   │   ├── model_comparison_sim.ipynb:  model comparison notebook for inference runs done on simulated data
│   │   ├── plotting_model1.py: python script to generate analysis figures of inference runs done on experimental and simulated data using model1
│   │   ├── plotting_model2.py: see above
│   │   └── plotting_model3.py: see above
│   └── cyto_ER_shuttling: follows the same format as the "ca_uptake" dir
│       ├── model_comparison_expt.ipynb
│       ├── model_comparison_sim.ipynb
│       ├── plotting_model1.py
│       ├── plotting_model2.py
│       └── plotting_model3.py
├── code
│   ├── run_pymc.py: python script to run PyMC with the specified model and parameters
│   └── sma.py: python script to perform smooth moving averaging
├── data
├── DE: folder that contains results of Differential Evolution
│   ├── ca_uptake
│   │   ├── analysis.ipynb: analysis notebook to generate analysis figures
│   │   ├── fit_results_expt_0.csv: fitting results of the first model parameter from 500 runs on the experimental data (see model1.py for parameter order)
│   │   ├── fit_results_expt_1.csv: see above
│   │   ├── fit_results_expt_2.csv: see above
│   │   ├── fit_results_expt_3.csv: see above
│   │   ├── fit_results_expt_4.csv: see above
│   │   ├── fit_results_sim_0.csv: fitting results of the first model parameter from 500 runs on the simulated data (see model1.py for parameter order)
│   │   ├── fit_results_sim_1.csv: see above
│   │   ├── fit_results_sim_2.csv: see above
│   │   ├── fit_results_sim_3.csv: see above
│   │   ├── fit_results_sim_4.csv: see above
│   │   └── model1.py: python script to perform DE using model1
│   └── cyto_ER_shuttling: follows the same format as the "ca_uptake" dir
│       ├── analysis.ipynb
│       ├── fit_results_expt_0.csv
│       ├── fit_results_expt_1.csv
│       ├── fit_results_expt_2.csv
│       ├── fit_results_sim_0.csv
│       ├── fit_results_sim_1.csv
│       ├── fit_results_sim_2.csv
│       └── model1.py
├── LICENSE
├── model: PyMC models
│   ├── ca_uptake
│   │   ├── model1
│   │   │   ├── model1.py: python script to perform MCMC sampling using model1
│   │   │   ├── params_expt.yaml: yaml file to define processing options of the input data, priors of the model parameters, and MCMC sampling for the experimental data
│   │   │   └── params_sim.yaml: see above
│   │   ├── model2
│   │   │   ├── model2.py: python script to perform MCMC sampling using model1
│   │   │   ├── params_expt.yaml: yaml file to define processing options of the input data, priors of the model parameters, and MCMC sampling for the experimental data
│   │   │   └── params_sim.yaml: see above
│   │   └── model3
│   │       ├── model3.py: python script to perform MCMC sampling using model1
│   │       ├── params_expt.yaml: yaml file to define processing options of the input data, priors of the model parameters, and MCMC sampling for the experimental data
│   │       └── params_sim.yaml: see above
│   └── cyto_ER_shuttling:  follows the same format as the "ca_uptake" dir
│       ├── model1
│       │   ├── model1.py
│       │   ├── params_expt.yaml
│       │   └── params_sim.yaml
│       ├── model2
│       │   ├── model2.py
│       │   ├── params_expt.yaml
│       │   └── params_sim.yaml
│       └── model3
│           ├── model3.py
│           ├── params_expt.yaml
│           └── params_sim.yaml
├── observed_data
│   ├── ca_uptake
│   │   ├── expt_data.csv: experimental observed data
│   │   └── sim_data.csv: simulated observed data 
│   └── cyto_ER_shuttling
│       ├── expt_data.csv: experimental observed data
│       └── sim_data.csv: simulated observed data
└── README.md
```

### Inference data
Due to the size limit, the PyMC inference data are available at Zenodo with DOI: 10.5281/zenodo.15177884

### To cite
Fang, X., Varughese, P., Osorio-Valencia, S., Zima, A.V., P.M. Kekenes-Huskey (2025). A Bayesian framework for systems model refinement and selection of calcium signaling

### Funding
Research reported in this publication was supported by the Maximizing Investigators’ Research Award (MIRA) (R35) from the National Institute of General Medical Sciences (NIGMS) of the National Institutes of Health (NIH) under grant number GM148284 to P.M.K.H. 
This work used Expanse at San Diego Supercomputer Center (SDSC) through allocation CHE140116 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.





