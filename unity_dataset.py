import os
import shutil
import time
import glob

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from osc_server import OscServer
from osc_client import OscClient


class UnityDataset(data.Dataset):
    def __init__(self, data_dir, img_size, num_images):
        super(UnityDataset, self).__init__()
        # Osc Server
        self.server = OscServer('127.0.0.1', 4444)
        self.server.activate(address='/done')
        # Osc Client
        self.client = OscClient('127.0.0.1', 3333)

        self.data_dir = data_dir
        self.directory_counter = 0

        self.num_images = num_images

        self.img_paths = glob.glob(f'{self.data_dir}/*')

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __del__(self):
        self.shutdown()

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

    def create_next_dataset(self, patch_path):
        self.data_dir = f'train_data_{self.directory_counter}'
        print(f'next data_dir: {self.data_dir}')
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        time.sleep(1)
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.client.send_string('/start', [self.data_dir, patch_path])
        print('Started to create dataset!')
        print(f'Next patch path : {patch_path}')

        # Receive message from Unity
        self.server.is_done = False
        print('waiting Unity process...')
        while not self.server.is_done:
            pass
        print('Done! Created next dataset')
        time.sleep(2)

        self.img_paths = glob.glob(f'{self.data_dir}/*')
        self.directory_counter = 1 - self.directory_counter

        print(f'num_images : {len(self.img_paths)}')

    def shutdown(self):
        self.server.shutdown()


if __name__ == "__main__":
    from tqdm import tqdm
    import torch.utils

    img_size = 416
    num_images = 614
    batch_size = 8

    dataset = UnityDataset(data_dir='train_data_0', img_size=img_size, num_images=num_images)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    epoch_length = len(train_loader)
    print(f'One epoch is {len(train_loader)}')

    img_paths = [
        'pics/0.png',
        'pics/qosmo.jpg'
    ]

    n_epochs = 2
    for epoch in range(n_epochs):
        path = img_paths[epoch]
        dataset.create_next_dataset(path)
        # print(path)
        for i_batch, img_batch in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=epoch_length):
            pass
