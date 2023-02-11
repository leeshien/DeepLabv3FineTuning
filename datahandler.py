from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from segdataset import SegmentationDataset


def get_dataloader_sep_folder(train_data_dir: str,
                              test_data_dir: str,
                              image_folder: str = 'Image',
                              mask_folder: str = 'Mask',
                              batch_size: int = 4,
                              input_size = [480,480]):
    """ Create Train and Test dataloaders from two
        separate Train and Test folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Image
        ---------Image1
        ---------ImageN
        ------Mask
        ---------Mask1
        ---------MaskN
        --Test
        ------Image
        ---------Image1
        ---------ImageM
        ------Mask
        ---------Mask1
        ---------MaskM

    Args:
        data_dir (str): The data directory or root.
        image_folder (str, optional): Image folder name. Defaults to 'Image'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Mask'.
        batch_size (int, optional): Batch size of the dataloader. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose(
      [
        transforms.ToTensor(),
        transforms.Resize(input_size)
      ]
    )


    image_datasets = {
        tag: SegmentationDataset(root=Path(data_dir) / tag,
                               transforms=data_transforms,
                               image_folder=image_folder,
                               mask_folder=mask_folder)
        for tag in zip(['Train', 'Test'], [train_data_dir, test_data_dir])
    }
    dataloaders = {
        tag: DataLoader(image_datasets[tag],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8,
                      drop_last=True)
        for tag in zip(['Train', 'Test'], [train_data_dir, test_data_dir])
    }
    return dataloaders


def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str = 'Images',
                                 mask_folder: str = 'Masks',
                                 fraction: float = 0.2,
                                 batch_size: int = 4):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.

    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose(
      [
        transforms.ToTensor(),
        transforms.Resize(input_size)
      ]
    )

    image_datasets = {
        tag: SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               seed=100,
                               fraction=fraction,
                               subset=tag,
                               transforms=data_transforms)
        for tag in ['Train', 'Test']
    }
    dataloaders = {
        tag: DataLoader(image_datasets[tag],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8,
                      drop_last=False)
        for tag in ['Train', 'Test']
    }
    return dataloaders
