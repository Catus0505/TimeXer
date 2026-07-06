import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from exp.exp_long_term_forecasting import (
    Exp_Long_Term_Forecast,
    compute_need_loss,
)
from models.cird import Model


def make_config(**overrides):
    values = dict(
        task_name="long_term_forecast",
        features="M",
        seq_len=24,
        pred_len=12,
        use_norm=1,
        patch_len=6,
        enc_in=3,
        d_model=8,
        n_heads=2,
        e_layers=1,
        d_ff=16,
        dropout=0.0,
        activation="gelu",
        num_need_slots=3,
        need_loss_weight=0.1,
        need_eps=1e-6,
        need_dim=8,
        q_dim=5,
        context_rank=2,
        cd_dim=8,
        cd_dropout=0.0,
        attention_temperature=1.0,
        null_supply=1,
        value_adapter_rank=4,
        cd_init_scale=0.1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def ci_parameters(model):
    yield from model.ci_embedding.parameters()
    yield from model.ci_encoder.parameters()
    yield from model.ci_head.parameters()


def has_nonzero_gradient(parameters):
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in parameters
    )


class CIRDModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = Model(make_config())
        self.x = torch.randn(2, 24, 3)
        self.target = torch.randn(2, 12, 3)

    def test_prediction_is_detached_ci_plus_cd_residual(self):
        prediction, aux = self.model(
            self.x,
            None,
            None,
            None,
            return_aux=True,
        )
        pure_ci, _ = self.model.forecast_ci(self.x)

        self.assertEqual(prediction.shape, (2, 12, 3))
        self.assertEqual(aux["ci_prediction"].shape, (2, 12, 3))
        self.assertEqual(aux["q_need"].shape, (2, 3, 3, 5))
        self.assertEqual(aux["need_variance"].shape, (2, 3, 3))
        self.assertEqual(aux["cd_residual"].shape, (2, 12, 3))
        self.assertEqual(aux["cd_attention"].shape, (2, 3, 3, 3))
        self.assertEqual(aux["null_attention"].shape, (2, 3, 3))
        self.assertEqual(
            set(aux),
            {
                "ci_prediction",
                "q_need",
                "need_variance",
                "cd_residual",
                "cd_attention",
                "null_attention",
            },
        )
        torch.testing.assert_close(
            aux["ci_prediction"],
            pure_ci,
        )
        torch.testing.assert_close(
            prediction,
            aux["ci_prediction"] + aux["cd_residual"],
        )

        prediction_only = self.model(
            self.x,
            None,
            None,
            None,
        )
        self.assertIsInstance(prediction_only, torch.Tensor)
        self.assertEqual(prediction_only.shape, (2, 12, 3))

    def test_final_forecast_loss_cannot_update_ci(self):
        prediction = self.model(
            self.x,
            None,
            None,
            None,
        )

        F.mse_loss(prediction, self.target).backward()

        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in ci_parameters(self.model)
            )
        )
        self.assertTrue(
            has_nonzero_gradient(self.model.need_net.parameters())
        )
        self.assertTrue(
            has_nonzero_gradient(self.model.cd_branch.parameters())
        )

    def test_standalone_ci_loss_updates_only_ci(self):
        _, aux = self.model(
            self.x,
            None,
            None,
            None,
            return_aux=True,
        )

        F.mse_loss(
            aux["ci_prediction"],
            self.target,
        ).backward()

        self.assertTrue(
            has_nonzero_gradient(ci_parameters(self.model))
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in self.model.need_net.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in self.model.cd_branch.parameters()
            )
        )

    def test_combined_loss_updates_each_owned_path(self):
        prediction, aux = self.model(
            self.x,
            None,
            None,
            None,
            return_aux=True,
        )
        loss_ci = F.mse_loss(
            aux["ci_prediction"],
            self.target,
        )
        loss_cd = F.mse_loss(prediction, self.target)
        loss_need = compute_need_loss(
            self.target,
            aux["ci_prediction"],
            aux["need_variance"],
            num_need_slots=3,
            eps=1e-6,
        )
        loss = loss_ci + loss_cd + 0.1 * loss_need

        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertTrue(
            has_nonzero_gradient(ci_parameters(self.model))
        )
        self.assertTrue(
            has_nonzero_gradient(self.model.need_net.parameters())
        )
        self.assertTrue(
            has_nonzero_gradient(self.model.cd_branch.parameters())
        )

    def test_trainer_adds_ci_and_need_losses_to_final_mse(self):
        prediction, aux = self.model(
            self.x,
            None,
            None,
            None,
            return_aux=True,
        )
        forecast_loss = F.mse_loss(prediction, self.target)
        experiment = object.__new__(Exp_Long_Term_Forecast)
        experiment.args = make_config()

        actual = experiment._add_training_losses(
            forecast_loss,
            aux,
            self.target,
            f_dim=0,
        )
        expected = (
            forecast_loss
            + F.mse_loss(aux["ci_prediction"], self.target)
            + 0.1
            * compute_need_loss(
                self.target,
                aux["ci_prediction"],
                aux["need_variance"],
                num_need_slots=3,
                eps=1e-6,
            )
        )

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
