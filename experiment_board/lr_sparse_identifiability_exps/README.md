The results and the setting of the of the synthetic experiments can be found in the [Experiment report](https://api.wandb.ai/links/indibi_at_splab_msu/0z451h52). The hyper-parameter selection is done as reported in [HP Study](https://api.wandb.ai/links/indibi_at_splab_msu/jwd26wk1). The identifiability study in independent runs with a control setting using unsupervised metrics Chosen hyper-parameters are applied on repeated (independent) experiment runs with similar control settings.

- `snn_logn_experiments_wandb.ipynb` Notebook contains an example hyper-parameter search.
- `exp1.ipynb` Notebook where the repeated experiments based on the searched hyper-parameters are used.
- `exp_nyc_taxi.ipynb` and `nyc_taxi_exp.py` contain the experiment scripts for NYC Taxi Dataset and `optuna_gorpca_nyc.db` contains the optuna database with the results.
- `./configs/` contain the configuration files for the experiments and hyper-parmeter search configurations.

*Errata:*
We later noticed that the BIC calculation was erroneous, however the results of the experiments still apply as the score was consistent with the other metrics we have used to select the hyper-parameters.
- NYC Taxi Dataset Experiments:
The hyper-parameter search is done as in the notebook `