from pathlib import Path
import random

from torchvision.datasets import utils
from skimage import io

import numpy as np

TRAIN_BATCH_SIZE = 10000
TEST_BATCH_SIZE = 10000
TRAIN_TEST_SPLIT = 0.7


def pickle_mopath(zip_file: Path, is_url: bool = False, dest: Path = None):
    meta_file = Path('mopath.meta')
    train_batch_str = 'train_batch_{%d}'
    test_batch_str = 'test_batch_{%d}'

    if dest is None:
        dest = zip_file.parent / 'zip-extraction'
    else:
        dest = dest / 'zip-extraction'
    if is_url:
        utils.download_and_extract_archive(
            url=str(zip_file),
            filename='MOPath.zip',
            download_root=str(dest),
            md5=None)
    else:
        utils.extract_archive(from_path=str(zip_file),
                              to_path=str(dest),
                              remove_finished=False)

    classes = list()
    image_names = list()
    for class_dir in dest.iterdir():
        class_name = str(class_dir).split('/')[-1]
        classes.append(class_name)
        files = list()
        if class_dir.is_dir():
            for img in class_dir.iterdir():
                files.append(img)
        for i, img in enumerate(files):
            image_names.append((class_name, img))
    random.seed(20200624)
    random.shuffle(image_names)
    train_images_len = int(TRAIN_TEST_SPLIT * len(image_names))
    train_images_ref = image_names[:train_images_len]
    test_images_ref = image_names[train_images_len:]
    class_indexes = {_class: i for i, _class in enumerate(classes)}
    with meta_file.open() as f:
        for c, i in class_indexes.items():
            f.write(f"{i} {c}")

    def chunks(_list, chunk):
        for i in range(0, len(_list), chunk):
            yield _list[i:i + chunk]

    train_images = list()
    train_labels = list()
    for i, (class_name, img) in enumerate(chunks(train_images_ref)):
        train_images.append(io.imread(str(img)))
        train_labels.append(class_indexes[class_name])
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    test_images = list()
    test_labels = list()
    for i, (class_name, img) in enumerate(chunks(test_images_ref)):
        test_images.append(io.imread(str(img)))
        test_labels.append(class_indexes[class_name])
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
