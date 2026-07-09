from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
from glob import glob
import os
import pandas as pd
from utils.aug import ImgTrainTransform, ImgEvalTransform
from sklearn.model_selection import KFold
import numpy as np

class MyDataset (data.Dataset):
    """
    This is the standard way to implement a dataset pipeline in PyTorch. We need to extend the torch.utils.data.Dataset
    class and implement the following methods: __len__, __getitem__ and the constructor __init__
    """

    def __init__(self, imgs_path, mos_scores, labels, transform=None):
        """
        The constructor gets the images path and their respectively labels and meta-data (if applicable).
        In addition, you can specify some transform operation to be carry out on the images.

        It's important to note the images must match with the labels (and meta-data if applicable). For example, the
        imgs_path[x]'s label must take place on labels[x].

        Parameters:
        :param imgs_path (list): a list of string containing the image paths
        :param mos_scores (list) a list of mos for each image
        :param transform (torchvision.transforms.Compose): transform operations to be carry out on the images
        """

        super().__init__()
        self.imgs_path = imgs_path
        self.mos_scores = mos_scores
        self.labels = labels

        # if transform is None, we need to ensure that the PIL image will be transformed to tensor, otherwise we'll get
        # an exception
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()


    def __len__(self):
        """ This method just returns the dataset size """
        return len(self.imgs_path)


    def __getitem__(self, item):
        """
        It gets the image, mos_scores and meta-data (if applicable) according to the index informed in `item`.
        It also performs the transform on the image.

        :param item (int): an index in the interval [0, ..., len(img_paths)-1]
        :return (tuple): a tuple containing the image, its label and meta-data (if applicable)
        """

        image = Image.open(self.imgs_path[item]).convert("RGB")

        # Applying the transformations
        image = self.transform(image)

        img_id = self.imgs_path[item].split('/')[-1].split('.')[0]

        if self.mos_scores is None:
            mos_scores = []
        else:
            mos_scores = self.mos_scores[item]

        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        return image, mos_scores, labels, img_id


def get_data_loader (imgs_path, mos_scores, labels, transform=None, batch_size=30, shuf=True, num_workers=4,
                     pin_memory=True):
    """
    This function gets a list og images path, their mos_scores and meta-data (if applicable) and returns a DataLoader
    for these files. You also can set some transformations using torchvision.transforms in order to perform data
    augmentation. Lastly, params is a dictionary that you can set the following parameters:
    batch_size (int): the batch size for the dataset. If it's not informed the default is 30
    shuf (bool): set it true if wanna shuffe the dataset. If it's not informed the default is True
    num_workers (int): the number thread in CPU to load the dataset. If it's not informed the default is 0 (which


    :param imgs_path (list): a list of string containing the images path
    :param mos_scores (list): a list of mos_scores for each image
    :param transform (torchvision.transforms, optional): use the torchvision.transforms.compose to perform the data
    augmentation for the dataset. Alternatively, you can use the jedy.pytorch.utils.augmentation to perform the
    augmentation. If it's None, none augmentation will be perform. Default is None
    :param batch_size (int): the batch size. If the key is not informed or params = None, the default value will be 30
    :param shuf (bool): if you'd like to shuffle the dataset. If the key is not informed or params = None, the default
    value will be True
    :param num_workers (int): the number of threads to be used in CPU. If the key is not informed or params = None, the
    default value will be  4
    :param pin_memory (bool): set it to True to Pytorch preload the images on GPU. If the key is not informed or
    params = None, the default value will be True
    :return (torch.utils.data.DataLoader): a dataloader with the dataset and the chose params
    """

    dt = MyDataset(imgs_path, mos_scores, labels, transform)
    dl = data.DataLoader (dataset=dt, batch_size=batch_size, shuffle=shuf, num_workers=num_workers,
                          pin_memory=pin_memory)
    return dl

def get_data_loaders_pad_diqa(dataset_path, num_folders=5, norm_score_10=False, batch_size=30):

    imgs_paths = []
    mos_list = []
    labels_list = []
        
    df = pd.read_csv(os.path.join(dataset_path, "metadata.csv"))
    df_quality = pd.read_csv(os.path.join(dataset_path, "quality.csv"))

    id_imgs_paths = glob(os.path.join(dataset_path, "train", "*.png"))

    imgs_paths.extend(id_imgs_paths)

    amount_imgs = len(imgs_paths)

    for i in range(amount_imgs):

        img_path = imgs_paths[i]

        img_path = os.path.basename(img_path)
        
        condition = df['Filename'] == img_path
        mos = df[condition]['general_perception']

        if norm_score_10:
            mos = (mos - 1)*2.25 + 1
        label = df_quality[condition]['quality_label'].astype(int)

        mos_list.append(mos.item())
        labels_list.append(label.item())

    boas = len([x for x in labels_list if x == 1])
    ruins = len([x for x in labels_list if x == 0])

    print(f"Amount of good quality images: {boas}")
    print(f"Amount of ruined quality images: {ruins}")

    train_dataloader_list, val_dataloader_list = [], []

    kf = KFold(n_splits=num_folders, shuffle=True, random_state=10)

    imgs_paths = np.array(imgs_paths)
    mos_list = np.array(mos_list)
    labels_list = np.array(labels_list)

    for i, (train_index, val_index) in enumerate(kf.split(imgs_paths)):

        train_imgs_paths, val_imgs_paths = imgs_paths[train_index], imgs_paths[val_index]
        train_mos, val_mos = mos_list[train_index], mos_list[val_index]
        train_labels, val_labels = labels_list[train_index], labels_list[val_index]

        train_dataloader = get_data_loader(train_imgs_paths, train_mos, train_labels, transform=ImgTrainTransform(), batch_size=batch_size, shuf=False)
        val_dataloader = get_data_loader(val_imgs_paths, val_mos, val_labels, transform=ImgEvalTransform(), batch_size=batch_size)

        train_dataloader_list.append(train_dataloader)
        val_dataloader_list.append(val_dataloader)

        print(f"Fold {i+1}:")
        print(f'Train: {len(train_imgs_paths)}, {len(train_labels)}')
        print(f'Validacao: {len(val_imgs_paths)}, {len(val_labels)}')

    return train_dataloader_list, val_dataloader_list

def get_test_data_loader_pad_diqa(dataset_path=None, ood_path=None, norm_score_10=False, batch_size=30):

    imgs_paths = []
    mos_list = []
    labels_list = []

    if dataset_path is not None:
        
        df = pd.read_csv(os.path.join(dataset_path, "metadata.csv"))
        df_quality = pd.read_csv(os.path.join(dataset_path, "quality.csv"))

        id_imgs_paths = glob(os.path.join(dataset_path, "test", "*.png"))

        imgs_paths.extend(id_imgs_paths)

        amount_imgs = len(imgs_paths)

        for i in range(amount_imgs):

            img_path = imgs_paths[i]

            img_path = os.path.basename(img_path)
            
            condition = df['Filename'] == img_path
            mos = df[condition]['general_perception']
            
            if norm_score_10:
                mos = (mos - 1)*2.25 + 1
            label = df_quality[condition]['quality_label'].astype(int)

            mos_list.append(mos.item())
            labels_list.append(label.item())

    if ood_path is not None:
        ood_imgs_paths = glob(os.path.join(ood_path, "*"))
        ood_labels = [0] * len(ood_imgs_paths)
        ood_mos = [1.0] * len(ood_imgs_paths)

        imgs_paths.extend(ood_imgs_paths)
        mos_list.extend(ood_mos)
        labels_list.extend(ood_labels)

    test_dataloader = get_data_loader(imgs_path=imgs_paths, mos_scores=mos_list, labels=labels_list, transform=ImgEvalTransform(), batch_size=batch_size, shuf=False)

    return test_dataloader

def get_test_data_loader(dataset_path, batch_size=30):

    imgs_paths = glob(os.path.join(dataset_path, "*"))

    test_dataloader = get_data_loader(imgs_path=imgs_paths, mos_scores=None, labels=None, transform=ImgEvalTransform(), batch_size=batch_size, shuf=False)

    return test_dataloader