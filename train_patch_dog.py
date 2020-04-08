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
        print(self.config)
        print('========================================')

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().to(device) # TODO: Why eval?
        self.patch_applier = PatchApplier().to(device)
        self.patch_transformer = PatchTransformer().to(device)
        self.prob_extractor = MaxProbExtractor(16, 80, self.config).to(device) # 0 is person
        self.adaIN_style_loss = AdaINStyleLoss().to(device)

        self.writer = self.init_tensorboard(mode)

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

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        n_epochs = 10000
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        orig_img = self.read_image('imgs/AF_patch_mayuu_05_red.jpg').to(device)
        adv_patch_cpu.requires_grad_(True)
        self.save_patch(adv_patch_cpu, 0)

        dataset = UnityDataset(data_dir='train_data_0', img_size=img_size, num_images=614)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        best_det_loss = 1.0
        for epoch in range(1, n_epochs):
            ep_det_loss = 0
            ep_adaIN_loss = 0
            ep_loss = 0
            bt0 = time.time()
            dataset.create_next_dataset(f'pics/{epoch - 1}.png')
            for i_batch, img_batch in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    optimizer.zero_grad()

                    img_batch = img_batch.to(device)
                    adv_patch = adv_patch_cpu.to(device)
                    p_img_batch = F.interpolate(img_batch, (self.darknet_model.height, self.darknet_model.width))

                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)
                    adaIN_loss = self.adaIN_style_loss(adv_patch.unsqueeze(0), orig_img.unsqueeze(0).to(device)) * 0.001

                    det_loss = torch.mean(-max_prob)
                    loss = det_loss + adaIN_loss

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_adaIN_loss += adaIN_loss.detach().cpu().numpy()

                    loss.backward()
                    optimizer.step()
                    adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                    bt1 = time.time()
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/adaIN_loss', adaIN_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                        self.writer.add_image('training_images', torchvision.utils.make_grid(p_img_batch), iteration)
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del output, max_prob, det_loss, p_img_batch, adaIN_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_adaIN_loss = ep_adaIN_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            self.save_patch(adv_patch_cpu, epoch)

            if det_loss.detach().cpu().numpy() < best_det_loss:
                best_det_loss = det_loss.detach().cpu().numpy()
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                if not os.path.exists('pics'):
                    os.mkdir('pics')
                im.save('pics/best_{}_{}.png'.format(epoch, det_loss.detach().cpu().numpy()), quality=100)

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('ADAIN LOSS: ', ep_adaIN_loss)
                print('EPOCH TIME: ', et1-et0)
                del output, max_prob, det_loss, p_img_batch, adaIN_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

        self.writer.close()

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

    def save_patch(self, adv_patch_cpu, epoch):
        im = transforms.ToPILImage('RGB')(adv_patch_cpu)
        if not os.path.exists('pics'):
            os.mkdir('pics')
        im.save('pics/{}.png'.format(epoch), quality=100)


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)


    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


