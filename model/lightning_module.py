import lightning as L
import torch
import numpy as np

from collections import deque

from torch import Tensor, nn
from torchmetrics import MeanMetric, MetricCollection

from .config import NNUELightningConfig
from .model import NNUEModel
from .lambda_utils import LambdaController


def _get_parameters(layers: list[nn.Module], get_biases: bool = False):
    return [
        p
        for layer in layers
        for name, p in layer.named_parameters()
        if ("bias" in name) == get_biases and p.requires_grad
    ]


def calculate_sf_loss(scorenet, score, outcome, loss_params, actual_lambda):
    # convert the network and search scores to an estimate match result
    # based on the win_rate_model, with scalings and offsets optimized
    q = (scorenet - loss_params.in_offset) / loss_params.in_scaling
    qm = (-scorenet - loss_params.in_offset) / loss_params.in_scaling
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

    s = (score - loss_params.out_offset) / loss_params.out_scaling
    sm = (-score - loss_params.out_offset) / loss_params.out_scaling
    pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

    # blend that eval based score with the actual game outcome
    t = outcome

    pt = pf * actual_lambda + t * (1.0 - actual_lambda)

    # use a MSE-like loss function
    loss = torch.pow(torch.abs(pt - qf), loss_params.pow_exp)
    if loss_params.qp_asymmetry != 0.0:
        loss = loss * ((qf > pt) * loss_params.qp_asymmetry + 1)

    weights = 1 + (2.0**loss_params.w1 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), loss_params.w2)
    loss = (loss * weights).sum() / weights.sum()

    return loss


class NNUE(L.LightningModule):

    def __init__(
        self,
        config: NNUELightningConfig,
        max_epoch=None,
        num_batches_per_epoch=None,
        param_index=0,
        num_psqt_buckets=8,
        num_ls_buckets=8,
    ):
        super().__init__()

        self.model: NNUEModel = NNUEModel(
            config.features,
            config.model_config,
            num_psqt_buckets,
            num_ls_buckets,
        )
        self.config = config
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.param_index = param_index

        # lazy init so `resume_from_model` with config changes works correctly
        self.optimizer_wrapper = None

        # Adaptive gradient-norm clipping state (see configure_gradient_clipping).
        # Clips the ~(100 - grad_clip_percentile)% largest-norm steps; a safety net
        # against a single bad batch exploding the weights to NaN.
        self._gc_pct = float(config.grad_clip_percentile)
        self._gc_warmup = int(config.grad_clip_warmup)
        self._gc_hist = deque(maxlen=int(config.grad_clip_history)) if self._gc_pct > 0 else None
        self._gc_steps = 0
        self._gc_clipped = 0

        # Initialize the lambda controller
        self.lambda_scheduler = LambdaController()

        self.loss_metrics = MetricCollection(
            {
                "train_loss_epoch": MeanMetric(),
                "val_loss_epoch": MeanMetric(),
                "test_loss_epoch": MeanMetric(),
            }
        )

    # --- setup optimizers and training hooks ---
    def configure_optimizers(self):
        optimizer_config = self.config.optimizer_config
        self.optimizer_wrapper = optimizer_config.get_optimizer_wrapper()

        LRs = [optimizer_config.lr] * 10

        ft_wd = optimizer_config.ft_weight_decay
        dense_wd = optimizer_config.dense_weight_decay
        factorized_wd = optimizer_config.factorized_weight_decay

        train_params = [
            # Feature Transformer
            {
                "params": _get_parameters([self.model.input], get_biases=False),
                "lr": LRs[0],
                "weight_decay": ft_wd,
            },
            {
                "params": _get_parameters([self.model.input], get_biases=True),
                "lr": LRs[1],
                "weight_decay": 0.0,
            },
            # Dense Layer Stacks
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.weight],
                "lr": LRs[2],
                "weight_decay": factorized_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.bias],
                "lr": LRs[3],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.weight],
                "lr": LRs[4],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.bias],
                "lr": LRs[5],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.weight],
                "lr": LRs[6],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.bias],
                "lr": LRs[7],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.output.linear.weight],
                "lr": LRs[8],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.output.linear.bias],
                "lr": LRs[9],
                "weight_decay": 0.0,
            },
        ]

        return self.optimizer_wrapper.configure_optimizers(train_params)

    # --- adaptive gradient-norm clipping (replaces Lightning's fixed clip) ---
    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        # Disabled -> behave exactly like no clipping.
        if self._gc_pct <= 0:
            return

        params = [p for p in self.parameters() if p.grad is not None]
        if not params:
            return

        # Total L2 grad norm over all parameters (single-GPU; not DDP-reduced).
        total_norm = torch.norm(
            torch.stack([p.grad.detach().norm(2) for p in params]), 2
        )
        self._gc_steps += 1

        if not bool(torch.isfinite(total_norm)):
            # inf/NaN gradient -> drop this step's update entirely.
            for p in params:
                p.grad.detach().zero_()
            self._gc_clipped += 1
        else:
            tn = float(total_norm)
            thr = None
            if len(self._gc_hist) >= self._gc_warmup:
                thr = float(
                    np.percentile(
                        np.fromiter(self._gc_hist, dtype=np.float64, count=len(self._gc_hist)),
                        self._gc_pct,
                    )
                )
            self._gc_hist.append(tn)  # record AFTER computing the threshold
            if thr is not None and thr > 0.0 and tn > thr:
                scale = thr / (tn + 1e-6)
                for p in params:
                    p.grad.detach().mul_(scale)
                self._gc_clipped += 1

        if self._gc_steps % 500 == 0:
            try:
                is_zero = self.trainer.is_global_zero
            except Exception:
                is_zero = True
            if is_zero:
                rate = 100.0 * self._gc_clipped / max(1, self._gc_steps)
                print(
                    f"[grad-clip] step {self._gc_steps}: clipped {self._gc_clipped} "
                    f"({rate:.2f}%), last_norm={float(total_norm):.4g}",
                    flush=True,
                )

    # --- train / eval switch ---
    def train(self, mode: bool = True):
        retval = super().train(mode)

        if self.optimizer_wrapper is not None:
            if mode:
                self.optimizer_wrapper.switch_to_train(True)
            else:
                self.optimizer_wrapper.switch_to_eval()

        return retval

    def eval(self):
        return self.train(False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # --- lightning hooks ---
    def on_train_epoch_start(self):
        self.optimizer_wrapper.on_train_epoch_start(self)

    def on_train_epoch_end(self):
        self.optimizer_wrapper.on_train_epoch_end(self)
        self._log_epoch_end("train_loss_epoch")

    def on_validation_epoch_start(self):
        self.optimizer_wrapper.on_validation_epoch_start(self)

    def on_validation_epoch_end(self):
        self._log_epoch_end("val_loss_epoch")

    def on_test_epoch_start(self):
        self.optimizer_wrapper.on_test_epoch_start(self)

    def on_test_epoch_end(self):
        self._log_epoch_end("test_loss_epoch")

    def on_save_checkpoint(self, checkpoint):
        self.optimizer_wrapper.on_save_checkpoint(self, checkpoint)
        self.lambda_scheduler.on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        self.lambda_scheduler.on_load_checkpoint(self, checkpoint)

    def on_train_batch_start(self, batch, batch_idx):
        self.optimizer_wrapper.on_train_batch_start(self, batch, batch_idx)

    def _log_epoch_end(self, loss_type):
        self.log(
            f"{loss_type}",
            self.loss_metrics[f"{loss_type}"],
            prog_bar=False,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )

    # --- Training step implementation ---

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val_loss")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test_loss")

    def step_(self, batch: tuple[Tensor, ...], batch_idx, loss_type):
        _ = batch_idx  # unused, but required by pytorch-lightning

        (
            us,
            them,
            white_indices,
            black_indices,
            outcome,
            score,
            piece_count,
        ) = batch
        scorenet = (
            self.model(
                us,
                them,
                white_indices,
                black_indices,
                piece_count,
                self.config.use_fake_act_quantization,
                self.config.use_fake_weight_quantization
            )
        )

        scorenet = scorenet * self.model.quantization.nnue2score

        actual_lambda = self.lambda_scheduler(
            loss_params=self.config.loss_params,
            current_epoch=self.current_epoch,
            max_epoch=self.max_epoch,
            is_training=self.training,
            scorenet=scorenet
        )

        sf_loss = calculate_sf_loss(
            scorenet, score, outcome, self.config.loss_params, actual_lambda
        )

        self.loss_metrics[f"{loss_type}_epoch"].update(sf_loss)
        self.log(
            loss_type,
            sf_loss,
            prog_bar=False,
            sync_dist=False,
            on_epoch=False,
            on_step=True,
        )
        return sf_loss
