#!/bin/bash
#SBATCH --job-name=dropout_sweep
#SBATCH --output=dropout_sweep_%j.out
#SBATCH --error=dropout_sweep_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug

# Set resource limits to avoid OpenBLAS threading issues
ulimit -u 16384  # Increase max user processes
ulimit -s unlimited  # Set unlimited stack size

# OpenBLAS environment variables to control threading
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# Prevent excessive thread spawning
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

# Additional OpenBLAS settings
export OPENBLAS_VERBOSE=0
export OPENBLAS_MAIN_FREE=1

# Set working directory
cd /home/gjp1993/links/scratch/COMP6321_Project

# Load required modules for shared libraries  
module load StdEnv/2023
module load gcc/12.3

# Initialize conda properly
export CONDA_BASE_ENV="/home/gjp1993/miniconda3"
source $CONDA_BASE_ENV/etc/profile.d/conda.sh

# Activate ml_project environment
echo "Activating ml_project conda environment..."
conda deactivate 2>/dev/null || true
conda activate ml_project

# Verify environment activation
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Fix library path issues
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Clean Python environment variables to avoid conflicts
unset PYTHONPATH
export PYTHONNOUSERSITE=1

echo "Starting dropout sweep at $(date)"
echo "Running on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Run the Python script
echo "Running dropout_sweep.py..."
python dropout_sweep.py

echo "Dropout sweep completed at $(date)"
