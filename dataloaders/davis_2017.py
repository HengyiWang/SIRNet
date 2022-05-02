import cv2
#from __future__ import division
import os
import numpy as np
import torch

from dataloaders.helpers import *
from torch.utils.data import Dataset
from PIL import Image

class DAVIS2017(Dataset):

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS-2016',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None):

        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir  # The location of the Davis2016
        self.transform = transform  # In this one, it has the custom transforms
        self.meanval = meanval
        self.seq_name = seq_name

        if self.train:
            fname = 'train'
        else:
            fname = 'val'

        if self.seq_name is None:

            with open(os.path.join(db_root_dir, 'ImageSets/2017', fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
                    img_list.extend(images_path)
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels.extend(lab_path)
        else:
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
            img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            labels = list(map(lambda x: os.path.join('Annotations/480p/', str(seq_name), x), name_label))

            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(img_list))  # Make sure they are equal

        self.img_list = img_list
        self.labels = labels

        print('Done initializing ' + fname + ' Dataset')  # fname means train/validation

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)

        num_obj = gt.max()


        sample = {'image': img, 'gt': gt}

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)


        gt_oh = torch.zeros((gt.shape[0], gt.shape[1], num_obj))
        for o in range(num_obj):
            gt_oh[:, :, o] = (sample['gt'] == (o + 1))[0].float()

        sample['gt'] = gt_oh.permute(2, 0, 1)

        info = Image.open(os.path.join(self.db_root_dir, self.labels[0])).getpalette()

        return sample, info

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.labels[idx] is not None:
            mask_file = os.path.join(self.db_root_dir, self.labels[idx])
            gt = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)


        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            raise NotImplementedError('Not implemented')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    db_root_dir = ''

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    db_train = DAVIS2017(train=False, db_root_dir=db_root_dir, transform=transforms, seq_name='blackswan')