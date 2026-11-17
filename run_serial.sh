cd "$(cd "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
##conda activate /scratch/network/zl8336/.conda/envs/GPU_Python
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export NUM_ITERS=$1
export DIVERGENCE_THRESHOLD=$2
export SAMPLES_PER_AXIS=$3
export NUM_WORKERS=1
python src/main.py