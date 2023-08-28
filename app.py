#!/usr/bin/env python

from __future__ import annotations

import argparse
import torch
import gradio as gr

from Scenimefy.options.test_options import TestOptions
from Scenimefy.models import create_model
from Scenimefy.utils.util import tensor2im

from PIL import Image
import torchvision.transforms as transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()

TITLE = '''
        Scene Stylization with <a href="https://github.com/Yuxinn-J/Scenimefy">Scenimefy</a>
        '''
DESCRIPTION = '''
<div align=center>
<p> 
Gradio Demo for Scenimefy - a model transforming real-life photos into Shinkai-animation-style images. 
To use it, simply upload your image, or click one of the examples to load them.  
For best outcomes, please pick a natural landscape image similar to the examples below. 
Kindly note that our model is trained on 256x256 resolution images, using much higher resolutions might affect its performance. 
Read more in our <a href="https://arxiv.org/abs/2308.12968">paper</a>. 
</p>
</div>
'''
EXAMPLES = [['0.jpg'], ['1.png'], ['2.jpg'], ['3.png'], ['4.png'], ['5.png'], ['6.jpg'], ['7.png'], ['8.png']]
ARTICLE = r"""
If Scenimefy is helpful, please help to ⭐ the <a href='https://github.com/Yuxinn-J/Scenimefy' target='_blank'>Github Repo</a>. Thank you! 
🤟 **Citation**
If our work is useful for your research, please consider citing:
```bibtex
@inproceedings{jiang2023scenimefy,
  title={Scenimefy: Learning to Craft Anime Scene via Semi-Supervised Image-to-Image Translation},
  author={Jiang, Yuxin and Jiang, Liming and Yang, Shuai and Loy, Chen Change},
  booktitle={ICCV},
  year={2023}
}
```
🗞️ **License**
This project is licensed under <a rel="license" href="https://github.com/Yuxinn-J/Scenimefy/blob/main/LICENSE.md">S-Lab License 1.0</a>. 
Redistribution and use for non-commercial purposes should follow this license.
"""


model = None


def initialize():
    opt = TestOptions().parse()  # get test options
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
   
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    global model
    model = create_model(opt)      # create a model given opt.model and other options

    dummy_data = {
        'A': torch.ones(1, 3, 256, 256),
        'B': torch.ones(1, 3, 256, 256),
        'A_paths': 'upload.jpg'
    }

    model.data_dependent_initialize(dummy_data)
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.parallelize()
    return model


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((w, h), method)


def get_transform():
    method=Image.BICUBIC
    transform_list = []
    # if opt.preprocess == 'none':
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def inference(img):
    transform = get_transform()
    A = transform(img.convert('RGB')) # A.shape: torch.Size([3, 260, 460])
    A = A.unsqueeze(0) # A.shape: torch.Size([1, 3, 260, 460])
    
    upload_data = {
        'A': A,
        'B': torch.ones_like(A),
        'A_paths': 'upload.jpg'
    }

    global model
    model.set_input(upload_data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()
    return tensor2im(visuals['fake_B'])


def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('*** Now using %s.'%(args.device))
    
    global model 
    model = initialize()

    gr.Interface(
        inference, 
        gr.Image(type="pil", label='Input'),
        gr.Image(type="pil", label='Output').style(height=300),
        theme=args.theme, 
        title=TITLE,
        description=DESCRIPTION, 
        article=ARTICLE, 
        examples=EXAMPLES,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share
    )

if __name__ == '__main__':
    main()
