from config import working_dir,mesh,shard,num_iters,divergence_threshold,samples_per_axis,points_per_axis
import os
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt

x_min=-2.0
x_max=1.0
y_min=-1.5
y_max=1.5




with mesh:
    @partial(jax.jit, static_argnames=("num_iters", "divergence_threshold"))
    def iter(c: jnp.ndarray, num_iters=1, divergence_threshold=2.0,pad_mask=1):
        z = jnp.zeros_like(c, dtype=jnp.complex128)
        mask = jnp.ones_like(c, dtype=jnp.bool) & pad_mask
        def body_fn(val,_):
            z, mask = val
            z = jnp.where(mask, z**2 + c, jnp.nan)
            mask = jnp.where(mask, jnp.abs(z) <= divergence_threshold, False)
            return (z, mask), None
        (z, mask), _ = jax.lax.scan(body_fn, init=(z, mask), length=num_iters)
        num_cells_inside = jnp.sum(mask)
        return z, mask, num_cells_inside


def mandelbrot_area(num_iters: int, divergence_threshold: float):
    dx=(x_max-x_min)/samples_per_axis
    dy=(y_max-y_min)/samples_per_axis
    re_axis = x_min+dx/2+dx*jnp.arange(points_per_axis)   #may include padding points outside [x_min,x_max]
    im_axis = y_min+dy/2+dy*jnp.arange(points_per_axis)   #may include padding points outside [y_min,y_max]
    pad_mask = (re_axis[:, jnp.newaxis] <= x_max) & (im_axis[jnp.newaxis, :] <= y_max) & (re_axis[:, jnp.newaxis] >= x_min) & (im_axis[jnp.newaxis, :] >= y_min)
    c = jax.device_put(re_axis[:, jnp.newaxis] + 1.0j *im_axis[jnp.newaxis, :], shard)
    z, mask, num_cells_inside = iter(c, num_iters=num_iters, divergence_threshold=divergence_threshold, pad_mask=pad_mask)
    area=num_cells_inside*dx*dy
    return mask[:samples_per_axis, :samples_per_axis], area

mask, area=mandelbrot_area(num_iters=num_iters, divergence_threshold=divergence_threshold)   
#print("Mandelbrot area:", area)
plt.imshow(
    mask.T,
    cmap="Blues",
    extent=(x_min, x_max, y_min, y_max),
    vmin=0,
    vmax=1,
    alpha=0.5,
)
plt.title(f"Mandelbrot Set, Area = {area}, \n Samples per Axis = {samples_per_axis}, Divergence Threshold = {divergence_threshold},num_iters = {num_iters}")
plt.savefig(os.path.join(working_dir, "area_demo.png"))
print(area)