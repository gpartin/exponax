import jax.numpy as jnp
import pytest

import exponax as ex
from exponax.stepper import KleinGordon, Wave

L = 2 * jnp.pi
PI = jnp.pi


# ===========================================================================
# Instantiation
# ===========================================================================


class TestKleinGordonInstantiation:
    @pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
    def test_instantiate(self, num_spatial_dims):
        stepper = KleinGordon(num_spatial_dims, 10.0, 25, 0.1)
        assert stepper.num_channels == 2
        assert stepper.num_spatial_dims == num_spatial_dims

    @pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
    def test_output_shape(self, num_spatial_dims):
        N = 16
        stepper = KleinGordon(num_spatial_dims, L, N, 0.01)
        u0 = jnp.zeros((2,) + (N,) * num_spatial_dims)
        u1 = stepper(u0)
        assert u1.shape == u0.shape
        assert jnp.all(jnp.isfinite(u1))

    def test_wrong_input_shape_raises(self):
        stepper = KleinGordon(1, L, 32, 0.01)
        with pytest.raises(ValueError, match="Expected shape"):
            stepper(jnp.zeros((1, 32)))  # needs 2 channels

    def test_default_params(self):
        stepper = KleinGordon(1, L, 32, 0.01)
        assert stepper.speed_of_sound == 1.0
        assert stepper.mass == 1.0

    def test_custom_params(self):
        stepper = KleinGordon(1, L, 32, 0.01, speed_of_sound=2.0, mass=3.0)
        assert stepper.speed_of_sound == 2.0
        assert stepper.mass == 3.0


# ===========================================================================
# mass=0 should recover Wave equation
# ===========================================================================


class TestKleinGordonRecoverWave:
    """When mass=0, KleinGordon must produce identical results to Wave."""

    @pytest.mark.parametrize("num_spatial_dims", [1, 2])
    def test_mass_zero_matches_wave(self, num_spatial_dims):
        N, dt, c = 32, 0.01, 1.5
        kg = KleinGordon(
            num_spatial_dims, L, N, dt, speed_of_sound=c, mass=0.0
        )
        wave = Wave(num_spatial_dims, L, N, dt, speed_of_sound=c)

        x = jnp.linspace(0, L, N, endpoint=False)
        if num_spatial_dims == 1:
            h0 = jnp.cos(2 * x)[None]
        else:
            h0 = jnp.cos(2 * x)[None, :, None] * jnp.ones((1, N, N))
        v0 = jnp.zeros_like(h0)
        u0 = jnp.concatenate([h0, v0], axis=0)

        u_kg = u0
        u_wave = u0
        for _ in range(20):
            u_kg = kg(u_kg)
            u_wave = wave(u_wave)

        assert u_kg == pytest.approx(u_wave, abs=1e-5)

    def test_mass_zero_matches_wave_multi_step(self):
        """Longer evolution to catch accumulation drift."""
        N, dt, c = 64, 0.005, 1.0
        kg = KleinGordon(1, L, N, dt, speed_of_sound=c, mass=0.0)
        wave = Wave(1, L, N, dt, speed_of_sound=c)

        x = jnp.linspace(0, L, N, endpoint=False)
        h0 = (jnp.cos(x) + 0.5 * jnp.cos(3 * x))[None]
        v0 = jnp.zeros_like(h0)
        u0 = jnp.concatenate([h0, v0], axis=0)

        u_kg = u0
        u_wave = u0
        for _ in range(100):
            u_kg = kg(u_kg)
            u_wave = wave(u_wave)

        assert u_kg == pytest.approx(u_wave, abs=1e-4)


# ===========================================================================
# Analytical correctness — 1D Klein-Gordon standing mode
# ===========================================================================


class TestKleinGordonAnalytical1D:
    """For h(x,0) = cos(k0 x), v(x,0) = 0:
    ω = sqrt(c²k0² + m²)
    h(x,t) = cos(k0 x) cos(ω t)
    v(x,t) = -ω cos(k0 x) sin(ω t)
    """

    def _make_stepper_and_ic(self, k0, c=1.0, m=1.0, N=64, dt=0.01):
        stepper = KleinGordon(1, L, N, dt, speed_of_sound=c, mass=m)
        x = jnp.linspace(0, L, N, endpoint=False)
        h0 = jnp.cos(k0 * x)[None]
        v0 = jnp.zeros_like(h0)
        u0 = jnp.concatenate([h0, v0], axis=0)
        omega = jnp.sqrt(c**2 * k0**2 + m**2)
        return stepper, x, u0, float(omega)

    @pytest.mark.parametrize("k0", [1, 2, 3, 5])
    def test_single_mode(self, k0):
        c, m, N, dt = 1.0, 2.0, 64, 0.01
        stepper, x, u0, omega = self._make_stepper_and_ic(k0, c, m, N, dt)

        n_steps = 10
        u = u0
        for _ in range(n_steps):
            u = stepper(u)
        t = n_steps * dt

        h_exact = jnp.cos(k0 * x) * jnp.cos(omega * t)
        v_exact = -omega * jnp.cos(k0 * x) * jnp.sin(omega * t)

        assert u[0] == pytest.approx(h_exact, abs=1e-4)
        assert u[1] == pytest.approx(v_exact, abs=1e-3)

    def test_mass_gap(self):
        """With k0=0 (uniform mode), oscillation is at ω = m (the mass gap)."""
        m, N, dt = 3.0, 32, 0.01
        stepper = KleinGordon(1, L, N, dt, speed_of_sound=1.0, mass=m)

        h0 = jnp.ones((1, N))  # k=0 mode
        v0 = jnp.zeros_like(h0)
        u0 = jnp.concatenate([h0, v0], axis=0)

        n_steps = 20
        u = u0
        for _ in range(n_steps):
            u = stepper(u)
        t = n_steps * dt

        h_exact = jnp.cos(m * t) * jnp.ones(N)
        assert u[0] == pytest.approx(h_exact, abs=1e-4)

    def test_energy_bounded(self):
        """Total energy should be conserved (bounded) over many steps."""
        k0, c, m, N, dt = 3, 1.0, 2.0, 64, 0.005
        stepper, x, u0, omega = self._make_stepper_and_ic(k0, c, m, N, dt)

        def energy(u):
            h, v = u[0], u[1]
            # KE + gradient PE + mass PE
            return jnp.sum(v**2 + c**2 * jnp.abs(jnp.fft.rfft(h))**2 + m**2 * h**2)

        e0 = energy(u0)
        u = u0
        for _ in range(200):
            u = stepper(u)
        e_final = energy(u)

        # Spectral solver should conserve energy to machine precision
        assert e_final == pytest.approx(float(e0), rel=1e-3)
