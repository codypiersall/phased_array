import numpy as np
import phased_array

def test_ula():
    d = 1.0
    n = 8
    λ = 2.0
    a_i = np.ones(n)
    arr = phased_array.PhasedArray.ula(d, n)
    x = np.linspace(-90, 90, 300)
    θ = np.radians(x)
    ϕ = np.zeros_like(θ)
    af = arr.array_factor(λ, a_i, θ, ϕ)
    assert af is not None
    # TODO: better test here

def test_planar():
    dx = 1.0
    dy = 1.0
    λ = 2.0
    nx = 8
    ny = 8
    a_i = np.ones(nx * ny)

    deg = np.linspace(-90, 90, 300)

    θ = np.radians(deg)
    ϕ = np.radians(deg)

    θθ, ϕϕ = np.meshgrid(θ, ϕ)

    array = phased_array.PhasedArray.planar(dx, dy, nx, ny)
    af = array.array_factor(λ, a_i, θθ, ϕϕ)

    assert af.shape == θθ.shape
