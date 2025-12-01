#!/bin/bash

# Default values
m_id=1
channels=(1 2 3 4 5 6 7 8)
metric="gic_1_two_phase"
extra_args=""
models=HoRPCA,SNN_LOGNTE,SNN_L1_GTV_T
time_limit="03:59:00"

# Argument parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --machine)
      m_id="$2"
      shift 2
      ;;
    --channels)
      IFS=',' read -r -a channels <<< "$2"
      shift 2
      ;;
    --metric)
      metric="$2"
      shift 2
      ;;
    --model-list)
      models="$2" #   IFS=',' read -r -a models <<< "$2"
      shift 2 #   shift 2
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
for ch_id in "${channels[@]}"; do
  sbatch <<EOF
#!/bin/bash --login
#SBATCH --job-name=${m_id}_${ch_id}_${metric}_${models}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G
#SBATCH --constraint=v100|h200
#SBATCH --output=results/slurm_outputs/%x_%j.out
#SBATCH --error=results/slurm_outputs/%x_%j.err
#SBATCH --time=${time_limit}

# Load necessary modules
module purge
module load CUDA/12.6.0
module load Conda/3

# Activate the conda environment
conda activate ml_gsp

# Change to the specified directory
cd /mnt/home/indibimu/repos/ML_GSP/experiment_board/smd_anomaly_detection

# Execute the Python script with arguments
python smd_experiment_runner.py ${models} --machine_id ${m_id} --channel_id ${ch_id} --metric ${metric} ${extra_args}
EOF
done