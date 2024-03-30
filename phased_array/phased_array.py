class Element:
    """An array element"""
    def __init__(self, x, y, z, pattern):
        self.x = x
        self.y = y
        self.z = z
        self.pattern = pattern

class PhasedArray:
    """
    A completely arbitrary phased array antenna.

    .. math::

        a = 4

    """

    def __init__(self, elements):
        self._elements = elements
        self._element

    def array_factor(self, wavelength, weights, theta, phi):
        r"""Calculate the array factor of an array.

        Assumes embedded element pattern is an omnidirectional antenna. Uses equations
        in Phased Array Antenna Handbook, 3rd Edition.

        .. math::


            \textbf{F} = 4
            \tag{1.50}


        """
        if weights.shape != self._array_pos.shape:
            raise ValueError(
                f"Invalid weights for array of shape {self._array_pos.shape}"
            )
        λ = wavelength
        θ = theta
        ϕ = phi
