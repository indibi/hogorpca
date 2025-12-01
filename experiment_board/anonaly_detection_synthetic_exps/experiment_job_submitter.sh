#!/bin/bash

# Default values
group="fidelity_exp"
config="coherent_SL_rel_err_cmaes"
seed=85
extra_args=""
models=("SNN_L1_GTVS" "SNN_L1_GTVS_F" "HORPCA" "HORPCA_F" "SNN_LOGNSN" "SNN_LOGNSN_F")
time_limit="03:59:00"

# Argument parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      group="$2"
      shift 2
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --model-list)
      IFS=',' read -r -a models <<< "$2"
      shift 2
      ;;
    --time-limit)
      time_limit="$2"
      shift 2
      ;;
    *)  # Anything else gets passed to the Python script
      extra_args+="$1 "
      shift
      ;;
  esac
done

# Loop over each model and submit a job
for model in "${models[@]}"; do
  sbatch <<EOF
#!/bin/bash --login
#SBATCH --job-name=${group}_${config}_${model}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G
#SBATCH --constraint=v100|h200
#SBATCH --output=${group}/slurm_outputs/%x_%j.out
#SBATCH --error=${group}/slurm_outputs/%x_%j.err
#SBATCH --time=${time_limit}

# Load necessary modules
module purge
module load CUDA/12.6.0
module load Conda/3

# Activate the conda environment
conda activate ml_gsp

# Change to the specified directory
cd /mnt/home/indibimu/repos/ML_GSP/experiment_board/anomaly_detection_journal_exps

# Execute the Python script with arguments
python simulated_experiment_runner.py ${group} ${config} ${seed} --model ${model} ${extra_args}
EOF
done