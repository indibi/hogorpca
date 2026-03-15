# Real Dataset Anomaly Detection Experiments
<p align="right">  <b>Created:</b> 7/22/2025, <b>Last Updated: </b> 3/14/2026, <b>Author: </b> Mert Indibi</p>
This folder contains scripts used to perform anomaly detection experiments on Server Machine Dataset, Event Detection with 2018 NYC Yellow Taxi Records and Airport-Beach-Urban Hyperspectral Images.

---------------
## Datasets
The datasets below (except Hyperspectral images) are processed into tensor format with the scripts in `data/` folder within the repository.
1. [Server Machine Dataset Anomaly Detection [1]](https://github.com/NetManAIOps/OmniAnomaly)
    The Server Machine Dataset timeseries can be downloaded from the link provided to reproduce the images. The script in `data/server_machine_dataset.py` can be used to tensorize and process the timeseries and the labels if the dataset is downloaded into the folder `data/server_machine_dataset/`, with the subfolders `data/server_machine_dataset/test/`, `data/server_machine_dataset/test_label/` and `data/server_machine_dataset/interpretation_label/`.
2. [ABU Hyperspectral Image [2]](https://xudongkang.weebly.com/data-sets.html)
    The images can be downloaded from the link provided to reproduce the experiments. The provided script in `data/hsi_abu_dataset.py` should work if the images within `.mat` files (`abu-airport-1.mat`,...) are extracted into `data/hsi_abu` folder directly.
3. [2018 NYC Yellow Taxi Records [3]](www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
    We provided the pre-processed files for NYC Taxi dataset in the `data/nyc_taxi_data/` folder for reproducibility.


### Experiment Pipeline:
The general pipeline of the experiments follow the steps below
1. Tensorization of time-series (if not naturally in tensor format)
2. Spatial or Feature graph formation representing the domain of tensor modes.
3. Estimation of tensor rank and the noise level within the observations
4. Tensor decomposition of the data into Low-rank (Representing normal behavior) and Sparse (Representing the anomalous behavior) parts.
5. Scoring the anomaly detection performance using the decomposed sparse part as anomaly score.

The jupyter notebooks `smd_and_nyc_pipeline.ipynb` and `hsi_pipeline.ipynb` contain the scripts that follow the steps mentioned above. MATLAB Implementations of the models [TRPCA [5]](), [RTD:OITNN-L, RTD:OITNN-O, and RTD:TNN [4]](https://github.com/pingzaiwang/OITNN/tree/master) can be found in the `other_models` folder. The matlab scripts `SMD_OITNN_experiment.m` and `HSI_OITNN_experiment.m` contain the implementations of the models mentioned above on the datasets. The combined results of the proposed methods and the benchmark methods are merged and presented in `matlab_coordination.ipynb` notebook.

**Notes:**
- **Hyper-parameter selection:** We followed the theorems provided by the authors of the models using the same variance estimates as we used in our models. These variances are estimates for additive gaussian white noise. Optimal hyper-parameter selection is a challenging task and the results could likely be improved with more effort.
- **Hyperspectral Images:** We initally applied the same noise estimation method we used in SMD and NYC Experiments, but our estimation method proved not robust in this dataset. We think this may be related to a de-noising process applied to the images where the noisy spectral bands of the data were removed. Due to this, we set the $\mathrm{[SNN]+[LOGN+GTV]}$ hyper-parameters corresponding to the the GTV regularization $\lambda_{m}$ slightly differently. For the `RTD:OITNN-L`, `RTD:OITNN-O` and `RTD:TNN` models, we set the hyper-parameters similar to before using the estimated variance to `0.1`, which is very low when the scale of the values is taken into account.

`configs` folder contains `model_configs.yaml` file that is used to specify and store the proposed family of $\mathrm{[SNN]+[LOGN+GTV]}$ domain topology aware robust tensor decomposition models.

## References:
- [1] Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., & Pei, D. (2019, July). Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 2828-2837).
- [2] “ABU Hyperspectral Image.” Xudong Kang’s Homepage, xudongkang.weebly.com/data-sets.html. Accessed 20 Feb. 2026.
- [3] “TLC Trip Record Data.” TLC Trip Record Data - TLC, NYC Taxi & Limousine Commission, www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.
- [4] Wang, Andong, et al. "Robust tensor decomposition via orientation invariant tubal nuclear norms." Science China Technological Sciences 65.6 (2022): 1300-1317.
- [5] Lu, Canyi, et al. "Tensor robust principal component analysis with a new tensor nuclear norm." IEEE transactions on pattern analysis and machine intelligence 42.4 (2019): 925-938.
