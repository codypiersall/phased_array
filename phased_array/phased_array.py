import typing
import numpy as np

# a phase pattern; given (θ, ϕ), return the power in linear units of the emitted energy
Pattern = typing.Callable[[float, float], float]


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

        a_i = np.asarray(weights)
        λ = wavelength
        θ = np.asarray(theta)
        ϕ = np.asarray(phi)
        orig_shape = θ.shape
        θ = θ.ravel()
        ϕ = ϕ.ravel()
        a_i = a_i.ravel()

        u_0 = np.sin(θ) * np.cos(ϕ)
        v_0 = np.sin(θ) * np.sin(ϕ)
        k = 2 * np.pi / λ
        rhat = np.array([u_0, v_0, np.cos(θ)])
        ri_dot_rhat = self.positions.T.dot(rhat)
        phase = np.exp(1j * k * ri_dot_rhat)
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
    def planar(cls, d_x, d_y, n_x, n_y):
        """
        Construct a Phased Array with 2D, planar, linear spaced elements.
        """
        elements = []
        for i in range(n_x):
            for j in range(n_y):
                element = Element(i * d_x, j * d_y, 0)
                elements.append(element)
        return cls(elements)
