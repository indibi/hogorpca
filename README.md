# Grouped Outlier Robust HoRPCA
This repository organizes the Grouped Outlier Robust HoRPCA model implementations and anomaly detection experiments performed in the publications.

## Model Implementations
The implementations of the Higher-Order Grouped Outlier Robust PCA models can be found in `./src/models/lr_ssd/` folders. Within this folder,
- `snn__logn_gtv.py` contains $\mathrm{[SNN]-[LOGN+GTV]}$ model implementation
- `snn_logs.py` contains $\mathrm{[SNN]-[LOGN]}$ model implementation.

The Singleton implementation of Higher-order Robust PCA ($\mathrm{HoRPCA}$) model can be found in `./src/models/horpca/horpca_torch.py`


*Remark:* $\mathrm{[SNN]-[LOGN+GTV]}$ can be considered a generalization of $\mathrm{[SNN]-[LOGN]}$ and $\mathrm{HoRPCA}$ algorithms with model hyper-parameters and grouping chosen accordingly, we often called on the $\mathrm{[SNN]-[LOGN+GTV]}$ implementation with the corresponding settings for convenience instead of calling the `snn_logs.py` or 

## Experiment Results and Setup
Example applications, experiment and experiment processes can be found in `./experiment_board/` folder for the publications [1,2] under the following structre:

1. Higher-order Grouped Outlier Robust PCA with Graph Total Variation (SIGPRO)
    1. Synthetic Experiments:
        The details of the experiment, the hyper-parameter selection scripts and the results can be found in `./experiment_board/anomaly_detection_synthetic_exps/`.
    2. NYC Event Detection and SMD Anomaly Detection Experiments:
        Dataset loader python scripts and classes can be found in `./data/nyc_taxi_dataset.py` and `./data/server_machine_dataset.py` folders.
        Dataset pre-processing steps and experiment scripts along with the results can be found in `./experiment_board/anomaly_detection_real_data_exps/`. Specifically, `./experiment_board/anomaly_detection_real_data_exps/smd_and_nyc_pipeline.ipynb` holds the crucial data processing and anomaly detection pipeline.
2. Higher-order Grouped Outlier Robust PCA [SSP_2025](https://ieeexplore.ieee.org/abstract/document/11073198 "Higher-Order Grouped Outlier Robust PCA for Spatio-temporal Anomaly Detection")
    The experiment scripts and the configurations can be found in `./experiment_board/lr_sparse_identifiability_exps/`. Please refer to the `README.md` file in the folder. The synthetic experiment results for this publication are also organized in [Experiment report](https://api.wandb.ai/links/indibi_at_splab_msu/0z451h52)


## References

- [SIGPRO]: Indibi, M., Aviyente, S. (2025). “Spatio-temporal Anomaly Detection: A Regularized Robust Tensor Decomposition with Graph Total Variation and Grouped Sparsity”, Under revision at EURASIP Journal of Signal Processing
- [SSP_2025]: Indibi, M., & Aviyente, S. (2025, June). Higher-Order Grouped Outlier Robust PCA for Spatio-temporal Anomaly Detection. In 2025     IEEE Statistical Signal Processing Workshop (SSP) (pp. 176-180). IEEE.


## Contributors
Mert Indibi
