import sys
import time
from PIL import Image, ImageDraw
from utils import *
from darknet import Darknet

import cv2


def detect(cfgfile, weightfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = cv2.imread('249.jpg')
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    # Simen: niet nodig om dit in loop te doen?
    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    finish = time.time()

    # if i == 1:
    # print('Predicted in %f seconds.' % (finish-start))

    class_names = load_class_names(namesfile)
    img = plot_boxes(Image.fromarray(img), boxes, class_names=class_names)
    img = np.array(img)
    # cv2.imshow('Result', img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img)
    print(pil_img.mode)

    pil_img.save('save_pillow.jpg')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        detect(cfgfile, weightfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile')
