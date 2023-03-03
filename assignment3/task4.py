import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from torch import optim
import torchvision
from dataloaders import load_cifar10, load_cifar10_transfer
from trainer import Trainer
from task2 import ExampleModel
from typing import List


class Transfer(nn.Module):

    def __init__(self):
        """

        """
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
       
    


def create_plots(trainers: List[Trainer], names: List[str], task_name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    for trainer, name in zip(trainers, names):
        utils.plot_loss(trainer.train_history["loss"], label=f"Training loss {name}", npoints_to_average=10)
        utils.plot_loss(trainer.validation_history["loss"], label=f"Validation loss {name}")
        plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    for trainer, name in zip(trainers, names):
        utils.plot_loss(trainer.validation_history["accuracy"], label=f"Validation Accuracy {name}")
        plt.legend()
    plt.savefig(plot_path.joinpath(task_name))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 7
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4

    dataloaders_transfer = load_cifar10_transfer(batch_size)
    model_transfer = Transfer()
    trainer_transfer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_transfer,
        dataloaders_transfer
    )
    trainer_transfer.optimizer = optim.Adam(model_transfer.parameters(), lr=learning_rate)
    trainer_transfer.train()


    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots([trainer, trainer_transfer], ["task2_model", "task4_model"], "task4")

if __name__ == "__main__":
    main()