import os
from pathlib import Path
working_dir = Path(__file__).parent
import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
num_iters = int(os.getenv("NUM_ITERS", 1000))
divergence_threshold = float(os.getenv("DIVERGENCE_THRESHOLD", 2.0))
samples_per_axis = int(os.getenv("SAMPLES_PER_AXIS", 10))
num_workers = int(os.getenv("NUM_WORKERS", 1))
assert num_workers > 0, "Number of workers must be positive"
assert divergence_threshold > 0, "Divergence threshold must be positive"
assert samples_per_axis > 0, "Samples per axis must be positive"
assert num_iters >= 0, "Number of iterations must be non-negative"
assert num_workers <= os.cpu_count(), f"Number of workers cannot exceed number of CPU devices ({os.cpu_count()})"
jax.config.update('jax_num_cpu_devices', num_workers)
GLOBAL_DEVICE_COUNT = jax.device_count()
assert GLOBAL_DEVICE_COUNT == num_workers, f"Global device count ({GLOBAL_DEVICE_COUNT}) must match num_workers ({num_workers})"
#assert samples_per_axis%num_workers == 0, f"samples_per_axis ({samples_per_axis}) must be divisible by num_workers ({num_workers})"
points_per_axis=(samples_per_axis//num_workers+1)*num_workers   # pad the points to be divisible by num_workers
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
devices=jax.local_devices()
mesh = Mesh(np.array(devices), ('x',))
shard = NamedSharding(mesh, P('x'))

#print("Working directory:", working_dir)
#print("Number of iterations:", num_iters)
#print("Divergence threshold:", divergence_threshold)
#print("Samples per axis:", samples_per_axis)
#print("Number of workers:", num_workers)

