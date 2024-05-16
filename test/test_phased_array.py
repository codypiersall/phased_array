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


def test_θϕ_to_uv():
    θϕ_to_uv = phased_array.θφ_to_uv
    assert np.allclose(θϕ_to_uv(0, 0), (0, 0))
    assert np.allclose(θφ_to_uv(np.pi, 0), (0, 0))
    # easy points on circle
    assert np.allclose(θφ_to_uv(np.pi / 2, 0), (1, 0))
    assert np.allclose(θφ_to_uv(np.pi / 2, np.pi / 2), (0, 1))
    assert np.allclose(θφ_to_uv(np.pi / 2, np.pi / 2), (0, 1))
    assert np.allclose(θφ_to_uv(np.pi / 2, np.pi), (-1, 0))
    assert np.allclose(θφ_to_uv(np.pi / 2, -np.pi / 2), (0, -1))

    assert np.allclose(θφ_to_uv(np.pi / 4, -np.pi / 2), (0, -np.sqrt(2) / 2))


def test_uv_to_θϕ():
    uv_to_θφ = phased_array.uv_to_θϕ

    #  a few values that are totally on the uv plane
    assert np.allclose(uv_to_θφ(0, 0), (0, 0))
    assert np.allclose(uv_to_θφ(1, 0), (np.pi / 2, 0))
    assert np.allclose(uv_to_θφ(np.sqrt(2) / 2, np.sqrt(2) / 2), (np.pi / 2, np.pi / 4))
    assert np.allclose(uv_to_θφ(0, 1), (np.pi / 2, np.pi / 2))
    assert np.allclose(uv_to_θφ(-1, 0), (np.pi / 2, np.pi))
    assert np.allclose(uv_to_θφ(0, -1), (np.pi / 2, -np.pi / 2))

    assert np.allclose(uv_to_θφ(np.sqrt(2) / 2, 0), (np.pi / 4, 0))


def test_θφr_to_xyz():
    θφr_to_xyz = phased_array.θφr_to_xyz
    assert np.allclose(θϕr_to_xyz(0, 0, 1), (0, 0, 1))
    assert np.allclose(θφr_to_xyz(np.pi, 0, 1), (0, 0, -1))
    # easy points on circle
    assert np.allclose(θφr_to_xyz(np.pi / 2, 0, 1), (1, 0, 0))
    assert np.allclose(θφr_to_xyz(np.pi / 2, np.pi / 2, 1), (0, 1, 0))
    assert np.allclose(θφr_to_xyz(np.pi / 2, np.pi / 2, 1), (0, 1, 0))
    assert np.allclose(θφr_to_xyz(np.pi / 2, np.pi, 1), (-1, 0, 0))
    assert np.allclose(θφr_to_xyz(np.pi / 2, -np.pi / 2, 1), (0, -1, 0))

    assert np.allclose(
        θφr_to_xyz(np.pi / 4, -np.pi / 2, 1), (0, -np.sqrt(2) / 2, np.sqrt(2) / 2)
    )
