"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm

from load_data import *
from unity_dataset import UnityDataset
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision
import subprocess

import patch_config
import sys
import time


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device: {}'.format(device))


class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.config.patch_size = 600
        self.config.batch_size = 4
        print(self.config)
        print('========================================')

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().to(device) # TODO: Why eval?
        self.patch_applier = PatchApplier().to(device)
        self.patch_transformer = PatchTransformer().to(device)
        #
        # https://github.com/dwaithe/yolov2/blob/master/cfg/coco.names
        #
        # self.prob_extractor = MaxProbExtractor(0, 80, self.config).to(device)  # 0 is person
        self.prob_extractor = MaxProbExtractor(21, 80, self.config).to(device)  # 0 is person
        self.adaIN_style_loss = AdaINStyleLoss().to(device)
        self.content_loss = ContentLoss().to(device)
        self.total_variation = TotalVariation().to(device)

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        # img_size = self.darknet_model.height
        img_size = self.config.patch_size
        batch_size = self.config.batch_size
        n_epochs = 10000
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        # orig_img = self.read_image('C:/Users/rystylee/Desktop/unlabeled-22-09-13/banana_class_5.0_tv_0.5/172_-0.8353019279318971.png').to(device)
        # orig_img = self.read_image('C:/Users/rystylee/Desktop/unlabeled-22-09-13/bear_class_5.0_tv_0.5/178_-0.8371262217496896.png').to(device)
        orig_img = self.read_image('D:/dev/PyTorch/UNLABELED/pics/190_0.991754635200872_-0.5869993895485803.png')

        dataset = InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        et0 = time.time()
        best_det_loss = 1.0
        for epoch in range(1, 3):
            ep_det_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.to(device)
                    lab_batch = lab_batch.to(device)

                    adv_patch = orig_img.to(device)
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))
                    # p_img_batch = F.interpolate(img_batch, (self.darknet_model.height, self.darknet_model.width))

                    # img = p_img_batch[1, :, :,]
                    # img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()

                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)

                    det_loss = torch.mean(max_prob)
                    loss = det_loss

                    ep_det_loss += det_loss.detach().cpu().numpy()

                    bt1 = time.time()
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        # del output, max_prob, det_loss, p_img_batch, adaIN_loss, loss
                        del output, max_prob, det_loss, p_img_batch, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()

            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)

            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('EPOCH TIME: ', et1-et0)
                del output, max_prob, det_loss, p_img_batch,  loss
                torch.cuda.empty_cache()

            et0 = time.time()

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)

    trainer = PatchTrainer(sys.argv[1])
    trainer.train()


if __name__ == '__main__':
    main()
