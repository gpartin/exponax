import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from .._base_stepper import BaseStepper
from .._spectral import build_scaled_wavenumbers
from ..nonlin_fun import ZeroNonlinearFun


class KleinGordon(BaseStepper):
    mass: float
    speed_of_sound: float
    frequency: Float[Array, " 1 ... (N//2)+1"]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        speed_of_sound: float = 1.0,
        mass: float = 1.0,
    ):
        """
        Timestepper for the d-dimensional (`d ∈ {1, 2, 3}`) Klein-Gordon
        equation on periodic boundary conditions.

        In 1d, the Klein-Gordon equation is given by

        ```
            uₜₜ = c² uₓₓ - m² u
        ```

        with `c ∈ ℝ` being the wave speed and `m ∈ ℝ` being the mass
        parameter. This is the relativistic generalization of the wave
        equation, fundamental to quantum field theory and lattice field
        simulations.

        In higher dimensions:

        ```
            uₜₜ = c² Δu - m² u
        ```

        **Dispersion relation:** ω(k) = √(c²|k|² + m²)

        Unlike the wave equation (ω = c|k|), the Klein-Gordon equation has a
        **mass gap** — no modes with ω < m exist. The group velocity
        v_g = c²|k|/ω is always less than c (massive dispersion).

        Internally, the same diagonalization approach as the
        [`exponax.stepper.Wave`][] stepper is used, but with
        the Klein-Gordon dispersion relation.

        The second-order equation is rewritten as a first-order system:

        ```
            hₜ = v
            vₜ = c² Δh - m² h
        ```

        In Fourier space, each wavenumber k oscillates at frequency
        ω(k) = √(c²|k|² + m²). The system is diagonalized into
        forward/backward traveling modes that each evolve as a pure
        phase rotation.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions `d`.
        - `domain_extent`: The size of the domain `L`; in higher dimensions
            the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same.
        - `dt`: The timestep size `Δt` between two consecutive states.
        - `speed_of_sound` (keyword-only): The wave speed `c`. Default: `1.0`.
        - `mass` (keyword-only): The mass parameter `m`. Default: `1.0`.

        **Notes:**

        - The stepper is unconditionally stable, no matter the choice of
            any argument because the equation is solved analytically in Fourier
            space.
        - Setting `mass = 0.0` recovers the standard wave equation.
        - The factors `c Δt / L` and `m Δt` together affect the dynamics.
        """
        self.speed_of_sound = speed_of_sound
        self.mass = mass
        wavenumber_norm = jnp.linalg.norm(
            build_scaled_wavenumbers(
                num_spatial_dims=num_spatial_dims,
                domain_extent=domain_extent,
                num_points=num_points,
            ),
            axis=0,
            keepdims=True,
        )
        # Klein-Gordon dispersion: ω(k) = sqrt(c²|k|² + m²)
        self.frequency = jnp.sqrt(
            speed_of_sound**2 * wavenumber_norm**2 + mass**2
        )
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=2,
            order=0,
        )

    def _forward_transform(
        self, u_hat: Complex[Array, " 2 ... (N//2)+1"]
    ) -> Complex[Array, " 2 ... (N//2)+1"]:
        """Transform (h, v) into diagonalized Klein-Gordon wave modes."""
        h_hat, v_hat = u_hat[0:1], u_hat[1:2]
        # Scale height to match velocity units: w = iω h
        omega_guard = jnp.where(self.frequency == 0, 1.0, self.frequency)
        w_hat = 1j * omega_guard * h_hat

        # Orthonormal rotation into wave modes
        pos = (1 / jnp.sqrt(2)) * (w_hat + v_hat)
        neg = (1 / jnp.sqrt(2)) * (w_hat - v_hat)
        return jnp.concatenate([pos, neg], axis=0)

    def _inverse_transform(
        self, waves_hat: Complex[Array, " 2 ... (N//2)+1"]
    ) -> Complex[Array, " 2 ... (N//2)+1"]:
        """Transform diagonalized wave modes back into (h, v)."""
        pos, neg = waves_hat[0:1], waves_hat[1:2]
        # Inverse rotation
        w_hat = (1 / jnp.sqrt(2)) * (pos + neg)
        v_hat = (1 / jnp.sqrt(2)) * (pos - neg)

        # Undo scaling to recover height
        omega_guard = jnp.where(self.frequency == 0, 1.0, self.frequency)
        h_hat = w_hat / (1j * omega_guard)
        return jnp.concatenate([h_hat, v_hat], axis=0)

    def _build_linear_operator(
        self, derivative_operator: Complex[Array, " D ... (N//2)+1"]
    ) -> Complex[Array, " 2 ... (N//2)+1"]:
        val = 1j * self.frequency
        return jnp.concatenate(
            (
                val,
                -val,
            ),
            axis=0,
        )

    def _build_nonlinear_fun(
        self, derivative_operator: Complex[Array, " D ... (N//2)+1"]
    ) -> ZeroNonlinearFun:
        return ZeroNonlinearFun(self.num_spatial_dims, self.num_points)

    def step_fourier(
        self, u_hat: Complex[Array, " 2 ... (N//2)+1"]
    ) -> Complex[Array, " 2 ... (N//2)+1"]:
        """
        Advance the state by one timestep in Fourier space.

        Overrides the base method to wrap the ETDRK step with the
        forward/inverse diagonalization transforms.
        """
        waves_hat = self._forward_transform(u_hat)
        waves_hat_next = super().step_fourier(waves_hat)
        u_hat_next = self._inverse_transform(waves_hat_next)

        # At k=0 with m>0, the system is still diagonalizable (ω(0) = m ≠ 0),
        # so no special DC correction is needed. However, if m=0 we fall back
        # to the wave equation DC behavior.
        if self.mass == 0.0:
            h_dc_idx = (0,) + (0,) * self.num_spatial_dims
            v_dc_idx = (1,) + (0,) * self.num_spatial_dims
            u_hat_next = u_hat_next.at[h_dc_idx].add(self.dt * u_hat[v_dc_idx])

        return u_hat_next
