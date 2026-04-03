# --------------------------------
# TensorFlow implementation of EWC
# --------------------------------

import tensorflow as tf

class EWCTensorFlow:
    def __init__(self, model, lambda_ewc=500.0, fisher_decay=0.95):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_decay = fisher_decay

        self.fisher = {}
        self.means = {}

        self.trainable_vars = [ # exclude embeddings
            v for v in model.trainable_variables
            if "embedding" not in v.name.lower()
        ]

    def update(self, dataset, batch_size=32):
        """
        Empirical Fisher Information (using loss gradients).
        dataset: tf.data.Dataset or (X, y)
        """
        if isinstance(dataset, tuple):
            dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size)

        fisher_new = {v.name: tf.zeros_like(v) for v in self.trainable_vars}
        n_samples = 0

        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                preds = self.model(x_batch, training=False)
                y_batch = tf.expand_dims(y_batch, axis=-1)
                loss = tf.keras.losses.binary_crossentropy(y_batch, preds)

            grads = tape.gradient(loss, self.trainable_vars)

            batch_size_actual = tf.shape(x_batch)[0]
            n_samples += batch_size_actual

            for v, g in zip(self.trainable_vars, grads):
                if g is not None:
                    fisher_new[v.name] += tf.reduce_sum(tf.square(g), axis=0) # approximation of Fisher diagonal

        for v in self.trainable_vars:
            fisher_estimate = fisher_new[v.name] / tf.cast(n_samples, tf.float32)

            if v.name in self.fisher:
                self.fisher[v.name] = (
                    self.fisher_decay * self.fisher[v.name]
                    + fisher_estimate
                )
            else:
                self.fisher[v.name] = fisher_estimate

            self.means[v.name] = tf.identity(v)

    def penalty(self):
        if not self.fisher:
            return 0.0

        loss = 0.0
        for v in self.trainable_vars:
            if v.name in self.fisher:
                loss += tf.reduce_sum(
                    self.fisher[v.name] * tf.square(v - self.means[v.name])
                )
        return self.lambda_ewc * loss


class EWCModel(tf.keras.Model):
    def __init__(self, base_model, ewc_handler):
        super().__init__()
        self.base_model = base_model
        self.ewc_handler = ewc_handler

    def compile(self, optimizer, loss, metrics):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            main_loss = self.compiled_loss(y, y_pred)
            ewc_loss = self.ewc_handler.penalty()
            total_loss = main_loss + ewc_loss

        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)

        return {
            **{m.name: m.result() for m in self.metrics},
            "ewc_loss": ewc_loss
        }

    def call(self, x):
        return self.base_model(x)





# -----------------------------
# PyTorch implementation of EWC
# -----------------------------

import torch
from collections import defaultdict
from transformers import Trainer

class EWCTorch:
    def __init__(
        self,
        model,
        device,
        lambda_ewc=500.0,
        fisher_decay=0.95,
        exclude_bias=True
    ):
        """
        fisher_decay < 1.0 → Online EWC
        fisher_decay = 1.0 → EWC standard
        """
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.fisher_decay = fisher_decay

        self.params = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if exclude_bias and ("bias" in n or "LayerNorm" in n):
                continue
            self.params[n] = p

        self.fisher = defaultdict(lambda: torch.zeros(1, device=device))
        self.means = {}

    @torch.no_grad()
    def _save_means(self, model):
        for n, p in model.named_parameters():
            if n in self.params:
                self.means[n] = p.detach().clone()

    def update(self, model, dataloader):
        """
        Calcola Fisher del task corrente e aggiorna quella globale
        """
        model.eval()
        fisher_new = {n: torch.zeros_like(p) for n, p in self.params.items()}

        for batch in dataloader:
            model.zero_grad()

            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = model(**batch)

            if "labels" in batch: # "labels" are in batch
                loss = outputs.loss
            else: # compute loss manually
                log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=1)
                loss = torch.nn.functional.nll_loss(log_probs, batch["labels"])
                loss.backward()

            for n, p in model.named_parameters():
                if n in fisher_new and p.grad is not None:
                    fisher_new[n] += p.grad.detach() ** 2 # approximation of Fisher diagonal

        # media
        for n in fisher_new:
            fisher_new[n] /= len(dataloader)

            # Online update
            if n in self.fisher:
                self.fisher[n] = (
                    self.fisher_decay * self.fisher[n] +
                    fisher_new[n]
                )
            else:
                self.fisher[n] = fisher_new[n]

        self._save_means(model)

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.means[n]) ** 2).sum()
        return self.lambda_ewc * loss


class EWCTrainer(Trainer):
    def __init__(self, *args, ewc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc = ewc

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.ewc is not None and self.ewc.fisher is not None:
            loss += self.ewc.penalty(model)
        return (loss, outputs) if return_outputs else loss