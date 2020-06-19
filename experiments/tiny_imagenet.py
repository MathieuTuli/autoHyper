"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from pathlib import Path
from skimage import io

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image

import torch


class TinyImageNet(Dataset):
    """`TinyImageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``TinyImagenet`` exists
        train (bool, optional): If True, creates dataset for training, else
            for test
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that
            takes in the target and transforms it.
    """
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    # md5 = "d41d8cd98f00b204e9800998ecf8427e"
    training_file = 'train'
    test_file = 'test'
    val_file = 'test'
    classes = [
        'n01443537',  'n02094433',  'n02410509',  'n02841315',
        'n03255030',  'n03838899',  'n04259630',  'n04597913',
        'n01629819',  'n02099601',  'n02415577',  'n02843684',
        'n03355925',  'n03854065',  'n04265275',  'n06596364',
        'n01641577',  'n02099712',  'n02423022',  'n02883205',
        'n03388043',  'n03891332',  'n04275548',  'n07579787',
        'n01644900',  'n02106662',  'n02437312',  'n02892201',
        'n03393912',  'n03902125',  'n04285008',  'n07583066',
        'n01698640',  'n02113799',  'n02480495',  'n02906734',
        'n03400231',  'n03930313',  'n04311004',  'n07614500',
        'n01742172',  'n02123045',  'n02481823',  'n02909870',
        'n03404251',  'n03937543',  'n04328186',  'n07615774',
        'n01768244',  'n02123394',  'n02486410',  'n02917067',
        'n03424325',  'n03970156',  'n04356056',  'n07695742',
        'n01770393',  'n02124075',  'n02504458',  'n02927161',
        'n03444034',  'n03976657',  'n04366367',  'n07711569',
        'n01774384',  'n02125311',  'n02509815',  'n02948072',
        'n03447447',  'n03977966',  'n04371430',  'n07715103',
        'n01774750',  'n02129165',  'n02666196',  'n02950826',
        'n03544143',  'n03980874',  'n04376876',  'n07720875',
        'n01784675',  'n02132136',  'n02669723',  'n02963159',
        'n03584254',  'n03983396',  'n04398044',  'n07734744',
        'n01855672',  'n02165456',  'n02699494',  'n02977058',
        'n03599486',  'n03992509',  'n04399382',  'n07747607',
        'n01882714',  'n02190166',  'n02730930',  'n02988304',
        'n03617480',  'n04008634',  'n04417672',  'n07749582',
        'n01910747',  'n02206856',  'n02769748',  'n02999410',
        'n03637318',  'n04023962',  'n04456115',  'n07753592',
        'n01917289',  'n02226429',  'n02788148',  'n03014705',
        'n03649909',  'n04067472',  'n04465501',  'n07768694',
        'n01944390',  'n02231487',  'n02791270',  'n03026506',
        'n03662601',  'n04070727',  'n04486054',  'n07871810',
        'n01945685',  'n02233338',  'n02793495',  'n03042490',
        'n03670208',  'n04074963',  'n04487081',  'n07873807',
        'n01950731',  'n02236044',  'n02795169',  'n03085013',
        'n03706229',  'n04099969',  'n04501370',  'n07875152',
        'n01983481',  'n02268443',  'n02802426',  'n03089624',
        'n03733131',  'n04118538',  'n04507155',  'n07920052',
        'n01984695',  'n02279972',  'n02808440',  'n03100240',
        'n03763968',  'n04133789',  'n04532106',  'n09193705',
        'n02002724',  'n02281406',  'n02814533',  'n03126707',
        'n03770439',  'n04146614',  'n04532670',  'n09246464',
        'n02056570',  'n02321529',  'n02814860',  'n03160309',
        'n03796401',  'n04149813',  'n04540053',  'n09256479',
        'n02058221',  'n02364673',  'n02815834',  'n03179701',
        'n03804744',  'n04179913',  'n04560804',  'n09332890',
        'n02074367',  'n02395406',  'n02823428',  'n03201208',
        'n03814639',  'n04251144',  'n04562935',  'n09428293',
        'n02085620',  'n02403003',  'n02837789',  'n03250847',
        'n03837869',  'n04254777',  'n04596742',  'n12267677']

    def __init__(self, root, train=True, transform=None,
                 download=False):
        # super(TinyImageNet, self).__init__(
        #     root=root,
        #     transform=transform,
        #     target_transform=target_transform)
        self.train = train  # training set or test set
        self.root = root
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data = list()
        self.targets = list()
        self.load_data(data_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def raw_folder(self):
        return Path(self.root) / str(self) / 'raw'

    @property
    def processed_folder(self):
        return Path(self.root) / str(self) / 'processed'

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (self.processed_folder / self.training_file).exists() and \
            (self.processed_folder / self.test_file).exists()

    def load_data(self, data_file):
        class_idx = {_class: i for i, _class in enumerate(self.classes)}
        for class_folder in (self.processed_folder / data_file).iterdir():
            class_foler_name = str(class_folder).split('/')[-1]
            for img in class_folder.iterdir():
                self.data.append(io.imread(str(img)))
                self.targets.append(class_idx[class_foler_name])

    def download(self):
        """
        Download the TinyImageNet data if it doesn't exist in
            processed_folder already.
        """

        if self._check_exists():
            return

        self.raw_folder.mkdir(exist_ok=True, parents=True)
        self.processed_folder.mkdir(exist_ok=True, parents=True)

        # download files
        download_and_extract_archive(
            self.url,
            download_root=str(self.raw_folder),
            filename=str(self) + '.zip',
            remove_finished=True,
            # extract_root=str(self.raw_folder / str(self)),
            md5=None)

        # process and save as torch files
        print('Processing...')

        extracted_root = self.raw_folder / 'tiny-imagenet-200'
        for class_dir in (extracted_root / 'train').iterdir():
            for img in (class_dir / 'images').iterdir():
                new_img = Path(str(img).replace(
                    str(self.raw_folder),
                    str(self.processed_folder)).replace(
                        'images/', '').replace(
                            'tiny-imagenet-200/', ''))
                new_img.parent.mkdir(parents=True, exist_ok=True)
                img.rename(new_img)

        val_dict = dict()
        val_test_files = extracted_root / 'val'
        val_test_split = 0.5
        with (val_test_files / 'val_annotations.txt').open() as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]  # img name : class
        for i, img in enumerate((val_test_files / 'images').iterdir()):
            img_name = str(img).split('/')[-1]
            if i < val_test_split * len(val_dict):
                folder = (self.processed_folder / 'val' / val_dict[img_name])
                folder.mkdir(parents=True, exist_ok=True)
                img.rename(str(folder / img_name))
            else:
                folder = (self.processed_folder / 'test' / val_dict[img_name])
                folder.mkdir(parents=True, exist_ok=True)
                img.rename(str(folder / img_name))

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


if __name__ == "__main__":
    import torchvision.transforms as transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    trainset = TinyImageNet(
        root=str('.'), train=True, download=True,
        transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4,
        pin_memory=True)

    testset = TinyImageNet(
        root=str('.'), train=False, download=True,
        transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True)
