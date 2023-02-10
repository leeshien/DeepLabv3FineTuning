from pathlib import Path

import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data

import datahandler
from model import createDeepLabv3
from trainer import train_model


@click.command()
@click.option("--train-data-directory",
              required=True,
              help="Specify the main train data directory.")
@click.option("--test-data-directory",
              default='',
              help="[Optional] Specify the main test data directory.")
@click.option("--image-folder",
              required=True,
              help="Specify the image folder name.")
@click.option("--mask-folder",
              required=True,
              help="Specify the mask folder name.")
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory.")
@click.option(
    "--epochs",
    default=25,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
def main(train_data_directory, test_data_directory, image_folder, mask_folder, exp_directory, epochs, batch_size):
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3()
    model.train()
#     train_data_directory = Path(train_data_directory)
#     test_data_directory = Path(test_data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    # criterion = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # Create the dataloader
    if test_data_directory:
      dataloaders = datahandler.get_dataloader_sep_folder(
          train_data_directory, test_data_directory, image_folder, mask_folder, batch_size=batch_size)
    else:
      dataloaders = datahandler.get_dataloader_single_folder(
          train_data_directory, image_folder, mask_folder, batch_size=batch_size)
      
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    # Save the trained model
    torch.save(model, exp_directory / 'final.pt')


if __name__ == "__main__":
    main()
