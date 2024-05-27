import typing
import numpy as np

# a phase pattern; given (θ, ϕ), return the power in linear units of the emitted energy
Pattern = typing.Callable[[float, float], float]

c = 3e8


def uniform_pattern(theta: float, phi: float) -> float:
    """A uniform pattern; illuminates everything"""
    return 1.0


class Element:
    """An array element"""

    def __init__(
        self, x: float, y: float, z: float, pattern: Pattern = uniform_pattern
    ):
        self.x = x
        self.y = y
        self.z = z
        self.pattern = pattern


class PhasedArray:
    """
    A completely arbitrary phased array antenna.

    """

    def __init__(self, elements: list[Element]):
        self._elements = elements
        x = []
        y = []
        z = []
        patterns = []
        for e in elements:
            x.append(e.x)
            y.append(e.y)
            z.append(e.z)
            patterns.append(e.pattern)

        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.positions = np.array([self.x, self.y, self.z])

    def Δd_at_θϕ(self, θ, ϕ):
        """Δd distance traveled to each element with angle of arrival θ, ϕ"""
        # from PhasedArray Antenna Handbook, 3rd edititon
        θ = np.asarray(θ).ravel()
        ϕ = np.asarray(ϕ).ravel()
        u_0, v_0 = θϕ_to_uv(θ, ϕ)
        rhat = np.array([u_0, v_0, np.cos(θ)])
        ri_dot_rhat = self.positions.T.dot(rhat)
        return ri_dot_rhat

    def weights_at_θϕ(self, wavelength, θ, ϕ):
        """Return the complex weights required at each element to point at θ and ϕ."""
        # TODO Refactor so that this code isn't duplicated between here and array_factor
        Δd = self.Δd_at_θφ(θ, ϕ)
        k = 2 * np.pi / wavelength
        phase = np.exp(-1j * k * Δd)
        return phase

    def array_factor(self, wavelength, weights, theta, phi):
        r"""Calculate the array factor of an array.

        Assumes embedded element pattern is an omnidirectional antenna. Uses equations
        in Phased Array Antenna Handbook, 3rd Edition.

        .. math::

           \gdef\rhat{\pmb{\hat{r}}}
           \gdef\xhat{\pmb{\hat{x}}}
           \gdef\yhat{\pmb{\hat{y}}}
           \gdef\zhat{\pmb{\hat{z}}}

           F(θ, ϕ) = \sum a_i \exp(jk \pmb{r}_i \cdot \rhat)

        where

        .. math::

            \begin{align*}
            k         &= 2 \frac{π}{λ} & \text {wave number} \\
            \rhat_0   &= \xhat u_0 + \yhat v_0 + \zhat \cos θ_0 & \text{direction of oncoming wave} \\
            \pmb{r}_i &= \xhat x_i + \yhat y_i + \zhat z_i & \text{ the position of the $i$th element} \\
            u         &= \sin {θ} \cos {ϕ}  & \text{direction cosine $u$} \\
            v         &= \sin{θ} \sin{ϕ} & \text{direction cosine $v$}
            \end{align*}

        """
        if weights.size != self.positions.shape[-1]:
            raise ValueError(
                f"Invalid weights for array of shape {self.positions.shape}"
            )

        theta = np.asarray(theta)
        orig_shape = theta.shape
        Δd = self.Δd_at_θφ(theta, phi)

        k = 2 * np.pi / wavelength
        phase = np.exp(1j * k * Δd)

        a_i = np.asarray(weights).ravel()
        F = a_i.dot(phase)
        F.shape = orig_shape
        return F

    @classmethod
    def ula(cls, d, n):
        """Uniform Linear Array

        A 1D, linear phased array, with uniform spacing between elements
        Args:
            d: distance between elements
            n: number of elements
        """
        elements = []
        for i in range(n):
            element = Element(i * d, 0, 0)
            elements.append(element)
        return cls(elements)

    @classmethod
    def planar(cls, dx: float, dy: float, nx: int, ny: int):
        """
        Construct a Phased Array with 2D, planar, linear spaced elements.

        Args:
            dx: spacing between x elements in meters
            dy: spacing between y elements in meters
            nx: number of x elements
            ny: number of x elements
        """
        elements = []
        for i in range(nx):
            for j in range(ny):
                element = Element(i * dx, j * dy, 0)
                elements.append(element)
        return cls(elements)


def uv_to_θϕ(u, v):
    """Projection from uv plane to (θ, ϕ) angles"""
    θ = np.arcsin(np.sqrt(u**2 + v**2))
    ϕ = np.arctan2(v, u)
    return θ, ϕ


uv_to_theta_phi = uv_to_θφ


def θϕ_to_uv(θ, ϕ):
    """Projection from θ, ϕ to u,v"""
    sin_θ = np.sin(θ)
    u = sin_θ * np.cos(ϕ)
    v = sin_θ * np.sin(ϕ)
    return u, v


def theta_phi_to_uv(theta, phi):
    return θϕ_to_uv(theta, phi)


def θϕr_to_xyz(θ, ϕ, r):
    u, v = θϕ_to_uv(θ, ϕ)
    x = r * u
    y = r * v
    z = r * np.cos(θ)
    return x, y, z
