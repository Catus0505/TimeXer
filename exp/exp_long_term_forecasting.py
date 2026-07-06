from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import inspect
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def unpack_model_output(output):
    if isinstance(output, tuple):
        prediction, aux = output
        return prediction, aux
    return output, None


def compute_need_loss(
    batch_y,
    ci_prediction,
    need_variance,
    num_need_slots,
    eps,
):
    horizon = batch_y.shape[1]
    if horizon % num_need_slots != 0:
        raise ValueError(
            "Prediction horizon must be divisible by num_need_slots."
        )
    if ci_prediction.shape != batch_y.shape:
        raise ValueError(
            "ci_prediction must have the same shape as batch_y."
        )

    expected_variance_shape = (
        batch_y.shape[0],
        batch_y.shape[2],
        num_need_slots,
    )
    if need_variance.shape != expected_variance_shape:
        raise ValueError(
            "need_variance must have shape "
            f"{expected_variance_shape}, got {tuple(need_variance.shape)}."
        )

    residual = batch_y - ci_prediction.detach()
    block_length = horizon // num_need_slots
    residual_energy = torch.stack(
        [
            residual[
                :,
                slot * block_length : (slot + 1) * block_length,
                :,
            ]
            .square()
            .mean(dim=1)
            for slot in range(num_need_slots)
        ],
        dim=-1,
    )
    stabilized_variance = need_variance + eps
    return (
        residual_energy / stabilized_variance
        + torch.log(stabilized_variance)
    ).mean()


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.model_supports_aux = (
            'return_aux' in inspect.signature(model.forward).parameters
        )

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _forward_model(
        self,
        batch_x,
        batch_x_mark,
        dec_inp,
        batch_y_mark,
        request_aux=False,
    ):
        if request_aux and self.model_supports_aux:
            return self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                return_aux=True,
            )
        return self.model(
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
        )

    def _add_need_loss(self, forecast_loss, aux, batch_y, f_dim):
        need_loss_weight = getattr(
            self.args, 'need_loss_weight', 0.0
        )
        if (
            need_loss_weight <= 0
            or not isinstance(aux, dict)
            or 'ci_prediction' not in aux
            or 'need_variance' not in aux
        ):
            return forecast_loss

        ci_prediction = aux['ci_prediction'][
            :, -self.args.pred_len:, f_dim:
        ]
        need_variance = aux['need_variance'][:, f_dim:, :]
        need_loss = compute_need_loss(
            batch_y,
            ci_prediction,
            need_variance,
            getattr(self.args, 'num_need_slots', 4),
            getattr(self.args, 'need_eps', 1e-6),
        )
        return forecast_loss + need_loss_weight * need_loss

    def _add_ci_loss(self, forecast_loss, aux, batch_y, f_dim):
        if (
            not isinstance(aux, dict)
            or 'ci_prediction' not in aux
        ):
            return forecast_loss

        ci_prediction = aux['ci_prediction'][
            :, -self.args.pred_len:, f_dim:
        ]
        if ci_prediction.shape != batch_y.shape:
            raise ValueError(
                "Sliced ci_prediction must have the same shape as "
                "batch_y."
            )
        ci_loss = nn.functional.mse_loss(
            ci_prediction,
            batch_y,
        )
        return forecast_loss + ci_loss

    def _add_training_losses(
        self,
        forecast_loss,
        aux,
        batch_y,
        f_dim,
    ):
        loss = self._add_ci_loss(
            forecast_loss,
            aux,
            batch_y,
            f_dim,
        )
        return self._add_need_loss(
            loss,
            aux,
            batch_y,
            f_dim,
        )

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, _ = unpack_model_output(model_output)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                request_aux = self.model_supports_aux
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self._forward_model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                            request_aux=request_aux,
                        )
                        outputs, aux = unpack_model_output(model_output)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        forecast_loss = criterion(outputs, batch_y)
                        loss = self._add_training_losses(
                            forecast_loss, aux, batch_y, f_dim
                        )
                        train_loss.append(loss.item())
                else:
                    model_output = self._forward_model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                        request_aux=request_aux,
                    )
                    outputs, aux = unpack_model_output(model_output)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    forecast_loss = criterion(outputs, batch_y)
                    loss = self._add_training_losses(
                        forecast_loss, aux, batch_y, f_dim
                    )
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, _ = unpack_model_output(model_output)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if self.args.features == 'MS':
                        outputs = np.tile(outputs, [1, 1, batch_y.shape[-1]])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
