import unittest

import torch

from layers.cird_layers import CDSourceEncoder, ResidualCDBranch


def make_branch(null_supply=True):
    return ResidualCDBranch(
        seq_len=24,
        patch_len=6,
        pred_len=12,
        num_need_slots=3,
        q_dim=5,
        cd_dim=8,
        n_heads=2,
        context_rank=2,
        dropout=0.0,
        attention_temperature=0.7,
        null_supply=null_supply,
        value_adapter_rank=4,
        cd_init_scale=0.1,
        need_eps=1e-6,
    )


class CDSourceEncoderTest(unittest.TestCase):
    def test_encodes_each_channel_independently(self):
        torch.manual_seed(1)
        encoder = CDSourceEncoder(
            seq_len=24,
            patch_len=6,
            cd_dim=8,
            n_heads=2,
            dropout=0.0,
        )
        encoder.eval()
        x_history = torch.randn(2, 3, 24)

        source_features = encoder(x_history)
        changed_history = x_history.clone()
        changed_history[:, 0, :] += 10.0
        changed_features = encoder(changed_history)

        self.assertEqual(source_features.shape, (2, 3, 8))
        torch.testing.assert_close(
            source_features[:, 1:, :],
            changed_features[:, 1:, :],
        )


class ResidualCDBranchTest(unittest.TestCase):
    def test_returns_expected_shapes_and_normalized_attention(self):
        torch.manual_seed(2)
        branch = make_branch()
        x_history = torch.randn(2, 3, 24)
        q_need = torch.randn(2, 3, 3, 5)
        need_variance = torch.rand(2, 3, 3) + 0.1

        delta_cd, aux = branch(x_history, q_need, need_variance)

        self.assertEqual(delta_cd.shape, (2, 12, 3))
        self.assertEqual(aux["source_features"].shape, (2, 3, 8))
        self.assertEqual(aux["context_tokens"].shape, (2, 2, 8))
        self.assertEqual(aux["context_attention"].shape, (2, 2, 3))
        self.assertEqual(aux["read_attention"].shape, (2, 3, 2))
        self.assertEqual(aux["k_supply"].shape, (2, 3, 5))
        self.assertEqual(aux["cd_attention"].shape, (2, 3, 3, 3))
        self.assertEqual(aux["null_attention"].shape, (2, 3, 3))
        self.assertEqual(aux["r_cd"].shape, (2, 3, 3, 8))

        diagonal = torch.diagonal(
            aux["cd_attention"],
            dim1=1,
            dim2=3,
        )
        self.assertEqual(torch.count_nonzero(diagonal).item(), 0)
        torch.testing.assert_close(
            aux["cd_attention"].sum(dim=-1)
            + aux["null_attention"],
            torch.ones(2, 3, 3),
        )
        torch.testing.assert_close(
            aux["context_attention"].sum(dim=-1),
            torch.ones(2, 2),
        )
        torch.testing.assert_close(
            aux["read_attention"].sum(dim=-1),
            torch.ones(2, 3),
        )

    def test_null_is_the_only_supply_for_one_channel(self):
        branch = make_branch()
        x_history = torch.randn(2, 1, 24)
        q_need = torch.randn(2, 1, 3, 5)
        need_variance = torch.rand(2, 1, 3) + 0.1

        delta_cd, aux = branch(x_history, q_need, need_variance)

        self.assertEqual(
            torch.count_nonzero(aux["cd_attention"]).item(),
            0,
        )
        torch.testing.assert_close(
            aux["null_attention"],
            torch.ones(2, 1, 3),
        )
        self.assertEqual(torch.count_nonzero(delta_cd).item(), 0)

    def test_one_channel_without_null_is_rejected(self):
        branch = make_branch(null_supply=False)
        x_history = torch.randn(2, 1, 24)
        q_need = torch.randn(2, 1, 3, 5)
        need_variance = torch.rand(2, 1, 3) + 0.1

        with self.assertRaisesRegex(ValueError, "at least two channels"):
            branch(x_history, q_need, need_variance)

    def test_cd_gradients_do_not_enter_need_variance(self):
        torch.manual_seed(3)
        branch = make_branch()
        x_history = torch.randn(2, 3, 24)
        q_need = torch.randn(2, 3, 3, 5, requires_grad=True)
        need_variance = (
            torch.rand(2, 3, 3, requires_grad=True) + 0.1
        )
        need_variance.retain_grad()

        delta_cd, _ = branch(x_history, q_need, need_variance)
        delta_cd.square().mean().backward()

        self.assertIsNotNone(q_need.grad)
        self.assertGreater(torch.count_nonzero(q_need.grad).item(), 0)
        self.assertIsNone(need_variance.grad)
        source_weight = branch.source_encoder.value_embedding.weight
        self.assertIsNotNone(source_weight.grad)
        self.assertGreater(
            torch.count_nonzero(source_weight.grad).item(),
            0,
        )
        self.assertIsNotNone(branch.q_match.weight.grad)
        self.assertGreater(
            torch.count_nonzero(branch.q_match.weight.grad).item(),
            0,
        )

    def test_rejects_incompatible_source_attention_dimensions(self):
        with self.assertRaisesRegex(
            ValueError,
            "cd_dim must be divisible by n_heads",
        ):
            ResidualCDBranch(
                seq_len=24,
                patch_len=6,
                pred_len=12,
                num_need_slots=3,
                q_dim=5,
                cd_dim=7,
                n_heads=2,
                context_rank=2,
                dropout=0.0,
                attention_temperature=1.0,
                null_supply=True,
                value_adapter_rank=4,
                cd_init_scale=0.1,
                need_eps=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
