import torch
import lightning
from dataset import Dataset, Validation
from model import Model

class Classifier(lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = Model()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), 2e-4)

    def on_train_epoch_start(self):
        self.train_results = []

    def training_step(self, batch, batch_index):
        image, label = batch
        output = self.model(image)
        loss = torch.nn.functional.cross_entropy(output, label)
        with torch.no_grad():
            indices = torch.argmax(output, 1)
            self.train_results.append(torch.where(indices == label, 1.0, 0.0))
        self.log("loss", loss.item(), True)
        return loss

    def on_train_epoch_end(self):
        results = torch.concat(self.train_results)
        accuracy = torch.mean(results)
        self.log("train_accuracy", accuracy.item() * 100, True)

    def on_validation_epoch_start(self):
        self.val_results = []

    def validation_step(self, batch, batch_index):
        image, label = batch
        output = self.model(image)
        with torch.no_grad():
            indices = torch.argmax(output, 1)
            self.val_results.append(torch.where(indices == label, 1.0, 0.0))

    def on_validation_epoch_end(self):
        results = torch.concat(self.val_results)
        accuracy = torch.mean(results)
        self.log("val_accuracy", accuracy.item() * 100, True)

if __name__ == "__main__":
    train_data = torch.utils.data.DataLoader(
        dataset=Dataset(r"C:\Users\Hilmi\Documents\GitHub\tiny-imagenet\train"),
        batch_size=16,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=True)
    val_data = torch.utils.data.DataLoader(
        dataset=Validation(),
        batch_size=16,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        persistent_workers=True)
    trainer = lightning.Trainer(max_epochs=20)
    trainer.fit(
        model=Classifier(),
        train_dataloaders=train_data,
        val_dataloaders=val_data)