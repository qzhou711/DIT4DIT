"""Unit tests for noise schedule implementations.

Verifies:
- SqrtMinusEpsSchedule samples lie in (0, 1) and are not uniform.
- LogitNormalSchedule produces mean-centered samples.
- PowerLawSchedule skew is correct for power > 1.
- build_schedule factory resolves all registered schedules.
"""

import math
import pytest
import torch

from mimic_videos.core.noise_schedule import (
    SqrtMinusEpsSchedule,
    LogitNormalSchedule,
    PowerLawSchedule,
    UniformSchedule,
    build_schedule,
)


DEVICE = torch.device("cpu")
N = 8192  # sample size for distribution checks


class TestSqrtMinusEpsSchedule:
    def test_output_range(self):
        sch = SqrtMinusEpsSchedule()
        samples = sch.sample(N, DEVICE)
        assert (samples > 0).all(), "All samples must be > 0"
        assert (samples < 1).all(), "All samples must be < 1"

    def test_not_uniform(self):
        """The sqrt distribution should differ significantly from Uniform(0,1)."""
        sch = SqrtMinusEpsSchedule()
        samples = sch.sample(N, DEVICE)
        # Uniform mean ≈ 0.5; sqrt-minus-eps mean should be > 0.5 (skewed high)
        mean = samples.mean().item()
        assert mean > 0.55, (
            f"SqrtMinusEps mean={mean:.4f} should be > 0.55 (skewed toward 1). "
            "A uniform distribution would give ~0.5."
        )

    def test_output_shape(self):
        sch = SqrtMinusEpsSchedule()
        s = sch.sample(32, DEVICE)
        assert s.shape == (32,)

    def test_reproducible_on_seeded(self):
        sch = SqrtMinusEpsSchedule()
        torch.manual_seed(0)
        s1 = sch.sample(16, DEVICE)
        torch.manual_seed(0)
        s2 = sch.sample(16, DEVICE)
        assert torch.allclose(s1, s2)


class TestLogitNormalSchedule:
    def test_output_range(self):
        sch = LogitNormalSchedule()
        samples = sch.sample(N, DEVICE)
        assert (samples > 0).all()
        assert (samples < 1).all()

    def test_near_half_mean(self):
        """Default logit-normal (mean=0) should produce mean ≈ 0.5."""
        sch = LogitNormalSchedule(mean=0.0, std=1.0)
        samples = sch.sample(N, DEVICE)
        mean = samples.mean().item()
        assert abs(mean - 0.5) < 0.05, f"Logit-normal mean={mean:.4f} expected near 0.5"


class TestPowerLawSchedule:
    def test_power_gt_1_skews_high(self):
        """High power → samples near 1."""
        sch = PowerLawSchedule(power=4.0)
        samples = sch.sample(N, DEVICE)
        assert samples.mean().item() > 0.7, "Power=4 should produce high values"

    def test_power_1_is_uniform(self):
        """Power = 1 → uniform distribution."""
        sch = PowerLawSchedule(power=1.0)
        samples = sch.sample(N, DEVICE)
        mean = samples.mean().item()
        assert abs(mean - 0.5) < 0.05

    def test_invalid_power_raises(self):
        with pytest.raises(ValueError):
            PowerLawSchedule(power=0.0)


class TestFlowMatchingUtils:
    def test_interpolate_at_zero(self):
        """τ=0 → output = x0."""
        x0 = torch.ones(4, 3)
        eps = torch.zeros(4, 3)
        tau = torch.zeros(4)
        sch = UniformSchedule()
        out = sch.interpolate(x0, eps, tau)
        assert torch.allclose(out, x0)

    def test_interpolate_at_one(self):
        """τ=1 → output = eps."""
        x0 = torch.zeros(4, 3)
        eps = torch.ones(4, 3)
        tau = torch.ones(4)
        sch = UniformSchedule()
        out = sch.interpolate(x0, eps, tau)
        assert torch.allclose(out, eps)

    def test_velocity_target_shape(self):
        x0 = torch.randn(4, 16)
        eps = torch.randn(4, 16)
        sch = UniformSchedule()
        v = sch.velocity_target(x0, eps)
        assert v.shape == x0.shape

    def test_euler_integrate_zero_velocity(self):
        """With zero velocity, output should equal initial state."""
        sch = UniformSchedule()
        x = torch.randn(2, 8)
        result = sch.euler_integrate(lambda xt, t: torch.zeros_like(xt), x, num_steps=10)
        assert torch.allclose(result, x)


class TestBuildSchedule:
    @pytest.mark.parametrize("name", ["logit_normal", "sqrt_minus_eps", "uniform", "power_law"])
    def test_all_registered(self, name):
        kwargs = {"power": 2.0} if name == "power_law" else {}
        sch = build_schedule(name, **kwargs)
        s = sch.sample(8, DEVICE)
        assert s.shape == (8,)

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            build_schedule("nonexistent_schedule")
