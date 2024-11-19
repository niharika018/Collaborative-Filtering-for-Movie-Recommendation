import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self, dataset, model, optimizer, metrics, epochs=10, batch_size=32, device="cpu"
    ):
        self.dataset = dataset
        self.model = model.to(device)
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.metrics = metrics

        self.train_log = []
        self.test_log = []

    def train(self, evaluate=False, verbose=True, progressbar=True):
        self.model.train()
        self.model.zero_grad()

        for epoch in range(self.epochs):
            epoch_losses = []
            pbar = tqdm(
                self.dataset.train_generator(self.batch_size),
                dynamic_ncols=True,
                total=(
                    self.dataset.train_size // self.batch_size
                    + (1 if self.dataset.train_size % self.batch_size > 0 else 0)
                ),
                leave=False,
                disable=not progressbar,
            )

            for batch in pbar:
                self.optimizer.zero_grad()

                batch = torch.LongTensor(batch).to(self.device).to(self.device)

                preds = self.model(batch)
                loss = self.model.criterion(batch, preds)

                loss.backward()

                self.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_description("[{}] Loss: {:,.4f}\t".format(epoch, loss.item()))

            pbar.reset()
            mean_epoch_loss = np.mean(epoch_losses)
            if verbose:
                print(
                    "Epoch {}: Avg Loss/Batch {:<20,.6f}".format(epoch, mean_epoch_loss)
                )

            self.train_log.append(mean_epoch_loss)

            if evaluate:
                self.test(verbose=verbose, progressbar=progressbar)

    def test(self, verbose=True, progressbar=True):
        self.model.zero_grad()
        self.model.eval()

        preds = []
        with torch.no_grad():
            pbar = tqdm(
                self.dataset.test_generator(),
                total=self.dataset.test_size,
                leave=False,
                disable=not progressbar,
            )

            for uid, pos_iid, neg_iids in pbar:
                batch = list(zip([uid] * 101, neg_iids + [pos_iid], [0] * 100 + [1]))
                batch = torch.LongTensor(batch).to(self.device)

                preds.append(self.model.run_eval(batch).cpu().numpy())

            pbar.reset()

        compiled = {}
        for m in self.metrics:
            func = self.metrics[m][0]
            args = self.metrics[m][1]
            result = func(preds, **args)
            compiled[m] = result
            if verbose:
                print(f"{m}: {result}")

        self.test_log.append(compiled)
