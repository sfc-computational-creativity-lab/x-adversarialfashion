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
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).to(device) # 0 is person
        # self.prob_extractor = MaxProbExtractor(1, 80, self.config).to(device) # 1 is bicycle

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

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
        # adv_patch_cpu = self.generate_patch("gray")
        adv_patch_cpu = self.read_image('imgs/521.png')

        dataset = InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        et0 = time.time()
        best_det_loss = 1.0
        for epoch in range(1, 2):
            ep_det_loss = 0
            ep_adaIN_loss = 0
            ep_c_loss = 0
            ep_loss = 0
            bt0 = time.time()
            # dataset.create_next_dataset(f'pics/{epoch - 1}.png')
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)
                #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                # adv_patch = adv_patch_cpu.to(device)
                adv_patch = adv_patch_cpu.to(device)
                # adv_patch = adv_patch * orig_img
                adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                # img = p_img_batch[1, :, :,]
                # img = transforms.ToPILImage()(img.detach().cpu())
                # img.show()

                output = self.darknet_model(p_img_batch)
                max_prob = self.prob_extractor(output)

                det_loss = torch.mean(max_prob)

                ep_det_loss += det_loss.detach().cpu().numpy()

                bt1 = time.time()
                if i_batch + 1 >= len(train_loader):
                    print('\n')
                else:
                    # del output, max_prob, det_loss, p_img_batch, adaIN_loss, loss
                    del output, max_prob, det_loss,
                    torch.cuda.empty_cache()
                bt0 = time.time()

            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)

            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            if True:
                print('  EPOCH NR: ', epoch),
                #print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                #print('ADAIN LOSS: ', ep_adaIN_loss)
                #print('    C LOSS: ', ep_c_loss)
                #print('EPOCH TIME: ', et1-et0)
                del output, max_prob, det_loss, p_img_batch
                torch.cuda.empty_cache()

            et0 = time.time()

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

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

    # def save_patch(self, adv_patch_cpu, epoch):
    #     im = transforms.ToPILImage('RGB')(adv_patch_cpu)
    #     if not os.path.exists('pics'):
    #         os.mkdir('pics')
    #     im.save('pics/{}.png'.format(epoch), quality=100)

    def save_patch(self, adv_patch_cpu, epoch, ep_det_loss):
        im = transforms.ToPILImage('RGB')(adv_patch_cpu)
        if not os.path.exists('pics'):
            os.mkdir('pics')
        im.save('pics/{}_{}.png'.format(epoch, ep_det_loss), quality=100)



def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)


    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


