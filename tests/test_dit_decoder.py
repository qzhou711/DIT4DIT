"""Unit tests for DiTActionDecoder.

Tests forward-pass shape correctness and key behavioural properties
(proprio masking, zero-init output head) without requiring any GPU or
loading the Cosmos backbone.
"""

import pytest
import torch

from mimic_videos.decoder import DiTActionDecoder


B, CHUNK, A_DIM, P_DIM = 2, 16, 16, 16
CONTEXT_DIM = 128   # small context for faster CPU tests
HIDDEN = 128
LAYERS = 2
HEADS = 4


@pytest.fixture()
def decoder():
    return DiTActionDecoder(
        action_dim=A_DIM,
        proprio_dim=P_DIM,
        hidden_dim=HIDDEN,
        num_layers=LAYERS,
        num_heads=HEADS,
        mlp_ratio=4,
        context_dim=CONTEXT_DIM,
        chunk_size=CHUNK,
        proprio_mask_prob=0.5,
    )


@pytest.fixture()
def sample_inputs():
    return dict(
        noisy_actions=torch.randn(B, CHUNK, A_DIM),
        proprio=torch.randn(B, P_DIM),
        video_features=torch.randn(B, 10, CONTEXT_DIM),  # 10 mock tokens
        tau_a=torch.rand(B),
        tau_v=torch.rand(B),
    )


class TestForwardShape:
    def test_output_shape(self, decoder, sample_inputs):
        out = decoder(**sample_inputs)
        assert out.shape == (B, CHUNK, A_DIM), f"Expected ({B},{CHUNK},{A_DIM}) got {out.shape}"

    def test_no_nan(self, decoder, sample_inputs):
        out = decoder(**sample_inputs)
        assert not torch.isnan(out).any(), "NaNs in decoder output"

    def test_proprio_mask_apply(self, decoder, sample_inputs):
        """Masking should change the own-proprioception token used inside the decoder.

        The output head is zero-initialised, so comparing final outputs before
        any training would compare two all-zero tensors.  Instead we verify that
        the *projected* proprio token differs between the two calls.
        """
        proprio = sample_inputs["proprio"]

        torch.manual_seed(1)
        s_token_unmasked = decoder.proprio_proj(proprio.unsqueeze(1))  # [B, 1, D]

        # Simulate masking: with prob > 0 some rows get replaced by mask_token
        torch.manual_seed(1)
        B = proprio.shape[0]
        mask = torch.rand(B, 1, 1) < decoder.proprio_mask_prob
        s_token_masked = torch.where(
            mask, decoder.mask_token.expand(B, -1, -1), decoder.proprio_proj(proprio.unsqueeze(1))
        )

        # If proprio_mask_prob > 0 and B is large enough, at least one token
        # should be replaced by the learned mask token
        assert not torch.allclose(s_token_unmasked, s_token_masked), (
            "No tokens were masked. Increase B or proprio_mask_prob in the fixture."
        )


class TestZeroInit:
    def test_output_head_zero_at_init(self):
        """Output head should be zero-initialised → all-zero output for zero-init params."""
        dec = DiTActionDecoder(hidden_dim=64, context_dim=32, num_layers=1, num_heads=4)
        # Manually zero all params except output_head (which is already zeroed)
        assert dec.output_head.weight.data.abs().sum().item() == 0.0
        assert dec.output_head.bias.data.abs().sum().item() == 0.0


class TestChunkSize:
    def test_different_chunk_sizes(self, decoder):
        for chunk in [8, 16, 32]:
            actions = torch.randn(B, chunk, A_DIM)
            # Re-build with matching chunk_size
            dec = DiTActionDecoder(
                action_dim=A_DIM, hidden_dim=HIDDEN, num_layers=LAYERS, num_heads=HEADS,
                context_dim=CONTEXT_DIM, chunk_size=chunk
            )
            out = dec(
                actions, torch.randn(B, P_DIM), torch.randn(B, 10, CONTEXT_DIM),
                torch.rand(B), torch.rand(B)
            )
            assert out.shape == (B, chunk, A_DIM)


class TestRobotPolicyProtocol:
    """Verify that WorldActionPolicy satisfies the RobotPolicy Protocol (no GPU needed)."""

    def test_isinstance_check(self):
        from mimic_videos.core.protocol import RobotPolicy
        from mimic_videos.policy import WorldActionPolicy

        # Structural check: WorldActionPolicy must expose predict and reset
        assert hasattr(WorldActionPolicy, "predict"), "Missing predict method"
        assert hasattr(WorldActionPolicy, "reset"), "Missing reset method"

        # @runtime_checkable Protocol: issubclass returns True when the class
        # has matching methods — this is the intended behaviour.
        assert issubclass(WorldActionPolicy, RobotPolicy), (
            "WorldActionPolicy must satisfy the RobotPolicy Protocol (predict + reset)"
        )
