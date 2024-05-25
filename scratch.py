import vispy.plot as vp
import numpy as np
from phased_array import PhasedArray

dx = 1.0
dy = 1.0
λ = 2.0
nx = 8
ny = 8
a_i = np.ones((nx, ny))

deg = np.linspace(-90, 90, 30)

θ = np.radians(deg)
# ϕ = np.radians(deg)
ϕ = np.radians([0.0, 0.1])

θθ, ϕϕ = np.meshgrid(θ, ϕ)

array = PhasedArray.planar(dx, dy, nx, ny)
af = array.array_factor(λ, a_i, θθ, ϕϕ)

print(af)
fig2 = vp.Fig()
plot2 = fig2[0, 0]
af_db = 20 * np.log10(np.abs(af))
af_db -= np.max(af_db)
af_db[af_db < -100] = -100
print(af_db.shape)
plot2.surface(af_db, x=ϕ, y=θ)
fig2.show()
