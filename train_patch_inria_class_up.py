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
from torch import autograd, per_channel_affine
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
        # self.prob_extractor = MaxProbExtractor(78, 80, self.config).to(device) # 0 is person
        # self.prob_extractor = MaxProbExtractor(1, 80, self.config).to(device) # 1 is bicycle
        # self.prob_extractor = MaxProbExtractor(46, 80, self.config).to(device) # 46 is banana
        self.prob_extractor_person = MaxProbExtractor(0, 80, self.config).to(device) # 0 is person
        self.prob_extractor_bear = MaxProbExtractor(21, 80, self.config).to(device) # 21 is bear
        self.adaIN_style_loss = AdaINStyleLoss().to(device)
        self.content_loss = ContentLoss().to(device)
        self.total_variation = TotalVariation().to(device)

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

        # img_size = self.darknet_model.height
        img_size = self.config.patch_size
        batch_size = self.config.batch_size
        n_epochs = 10000
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")
        # adv_patch_cpu = self.read_image('imgs/01.png')
        orig_img = self.read_image('imgs/bear2.jpg').to(device)
        orig_img_style = self.read_image('imgs/210825_ダギング_6.jpg').to(device)

        adv_patch_cpu.requires_grad_(True)
        self.save_patch(adv_patch_cpu, 0, 0)

        dataset = InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        best_det_loss = 1.0
        for epoch in range(1, n_epochs):
            ep_det_loss_person = 0
            ep_det_loss_bear = 0
            ep_adaIN_loss = 0
            ep_c_loss = 0
            ep_loss = 0
            bt0 = time.time()
            # dataset.create_next_dataset(f'pics/{epoch - 1}.png')
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    optimizer.zero_grad()

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
                    max_prob_person = self.prob_extractor_person(output)
                    max_prob_bear = self.prob_extractor_bear(output)
                    # adaIN_loss = self.adaIN_style_loss(adv_patch.unsqueeze(0), orig_img.unsqueeze(0).to(device)) * 0.1
                    adaIN_loss = self.adaIN_style_loss(adv_patch.unsqueeze(0), orig_img_style.unsqueeze(0).to(device)) * 0.0
                    # adaIN_loss = 0.0
                    c_loss = self.content_loss(adv_patch, orig_img) * 5.0
                    tv_loss = self.total_variation(adv_patch) * 0.5

                    det_loss_person = torch.mean(max_prob_person)
                    det_loss_bear = torch.mean(-max_prob_bear)
                    # loss = det_loss + adaIN_loss
                    # loss = det_loss + adaIN_loss + torch.max(c_loss, torch.tensor(0.1).to(device))
                    # loss = det_loss + adaIN_loss + torch.max(c_loss, torch.tensor(0.1).to(device)) + torch.max(tv_loss, torch.tensor(0.1).to(device))
                    # loss = det_loss + adaIN_loss + c_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
                    # loss = det_loss + c_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
                    loss = det_loss_person + det_loss_bear + adaIN_loss + c_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))

                    ep_det_loss_person += det_loss_person.detach().cpu().numpy()
                    ep_det_loss_bear += det_loss_bear.detach().cpu().numpy()
                    ep_adaIN_loss += adaIN_loss.detach().cpu().numpy()
                    # ep_loss += loss
                    ep_c_loss += c_loss.detach().cpu().numpy()

                    loss.backward()
                    optimizer.step()
                    adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                    bt1 = time.time()
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        # self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss_person', det_loss_person.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss_bear', det_loss_bear.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/adaIN_loss', adaIN_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/content_loss', c_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                        self.writer.add_image('training_images', torchvision.utils.make_grid(p_img_batch), iteration)
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        # del output, max_prob, det_loss, p_img_batch, adaIN_loss, loss
                        # del output, max_prob, det_loss, p_img_batch, adaIN_loss, c_loss, loss
                        del output, max_prob_person, max_prob_bear, det_loss_person, det_loss_bear, p_img_batch, adaIN_loss, c_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()

            et1 = time.time()
            # ep_det_loss = ep_det_loss/len(train_loader)
            ep_det_loss_person = ep_det_loss_person/len(train_loader)
            ep_det_loss_bear = ep_det_loss_bear/len(train_loader)
            ep_adaIN_loss = ep_adaIN_loss/len(train_loader)
            ep_c_loss = ep_c_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            # self.save_patch(adv_patch_cpu, epoch, ep_det_loss)
            self.save_patch(adv_patch_cpu, epoch, str(ep_det_loss_person) + '_' + str(ep_det_loss_bear))

            # self.writer.add_image('patch', adv_patch_cpu, epoch)
            # self.writer.add_image('training_images', torchvision.utils.make_grid(p_img_batch), epoch)

            # if det_loss.detach().cpu().numpy() < best_det_loss:
            #     best_det_loss = det_loss.detach().cpu().numpy()
            #     im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #     if not os.path.exists('pics'):
            #         os.mkdir('pics')
            #     im.save('pics/best_{}_{}.png'.format(epoch, det_loss.detach().cpu().numpy()), quality=100)

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                # print('  DET LOSS: ', ep_det_loss)
                print('DET PERSON LOSS: ', ep_det_loss_person)
                print('DET PERSON BEAR: ', ep_det_loss_bear)
                print('ADAIN LOSS: ', ep_adaIN_loss)
                print('    C LOSS: ', ep_c_loss)
                print('EPOCH TIME: ', et1-et0)
                #im.save("saved_patches/patchnew1.jpg")
                # del output, max_prob, det_loss, p_img_batch, adaIN_loss, loss
                # del output, max_prob, det_loss, p_img_batch, adaIN_loss, c_loss, loss
                del output, max_prob_person, max_prob_bear, det_loss_person, det_loss_bear, p_img_batch, adaIN_loss, c_loss, loss
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


