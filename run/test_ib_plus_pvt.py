import torch
import os
import argparse
import tqdm
import sys

import torch.nn.functional as F
import numpy as np

from PIL import Image
from torch.nn import modules

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

# import sys
# sys.path.append('../utils')  # 或者你的utils文件夹的路径
# sys.path.append('../lib')  # 或者你的utils文件夹的路径

from utils.utils import *
# from utils.utils_pranet import *
from utils.dataloader import *
# from lib.PraNet_v3b_3_1 import PraNet_v3b_3_1
from lib.IBandEDL_plus_pvt import IBandEDL_plus_pvt
# from lib.ab_ff import ab_ff

def test(opt, args):
    model = eval(opt.Model.name)()
    model.load_state_dict(torch.load(os.path.join(
        opt.Test.Checkpoint.checkpoint_dir, 'latest_fold_5.pth')), strict=True)
        # opt.Test.Checkpoint.checkpoint_dir, 'best_epoch_145.pth')), strict=True)

    model.cuda()
    model.eval()

    if args.verbose is True:
        testsets = tqdm.tqdm(opt.Test.Dataset.testsets, desc='Total TestSet', total=len(
            opt.Test.Dataset.testsets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        testsets = opt.Test.Dataset.testsets

    for testset in testsets:
        data_path = os.path.join(opt.Test.Dataset.root, testset)
        save_path = os.path.join(opt.Test.Checkpoint.results_dir, testset)

        os.makedirs(save_path, exist_ok=True)

        test_dataset = eval(opt.Test.Dataset.type)(root=data_path, transform_list=opt.Test.Dataset.transform_list)

        test_loader = data.DataLoader(dataset=test_dataset,
                                        batch_size=1,
                                        num_workers=opt.Test.Dataloader.num_workers,
                                        pin_memory=opt.Test.Dataloader.pin_memory)

        if args.verbose is True:
            samples = tqdm.tqdm(test_loader, desc=testset + ' - Test', total=len(test_loader),
                                position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = test_loader


        with torch.no_grad():
            for sample in samples:
                sample = to_cuda(sample)
                # with torch.no_grad():
                out = model(sample, opt.Train.Dataloader.batch_size, opt.Train.Scheduler.epoch)
                # print(sample['shape'])
                
                sample_height = sample['shape'][0].item()  # 获取样本的高度
                sample_width = sample['shape'][1].item()   # 获取样本的宽度
                alpha_a = out['alpha_a']
                s = torch.sum(alpha_a, dim=1, keepdim=True)
                p = alpha_a / (s.expand(alpha_a.shape))

                pred = p[:,1]
                pred = pred.view(352, 352)
                pred = pred.detach().cpu().numpy()
                pred = np.array(pred)

                pred = cv2.resize(pred, (sample_width, sample_height))


            

                pred = (pred - pred.min()) / \
                (pred.max() - pred.min() + 1e-8)

                # 释放 GPU 内存
                torch.cuda.empty_cache()

            



                # out['pred'] = F.interpolate(
                #     out['pred'], sample['shape'], mode='bilinear', align_corners=True)

                # out['pred'] = out['pred'].data.cpu()
                # out['pred'] = torch.sigmoid(out['pred'])
                # out['pred'] = out['pred'].numpy().squeeze()
                # out['pred'] = (out['pred'] - out['pred'].min()) / \
                #     (out['pred'].max() - out['pred'].min() + 1e-8)
                
                # for j in range(out['pred'].shape[0]):
                #     image = ((out['pred'][j] > .5) * 255).astype(np.uint8)
                #     image_path = os.path.join(save_path, f"{sample['name'][j]}.png")
                #     Image.fromarray(image).save(image_path)

                Image.fromarray(((pred > .5) * 255).astype(np.uint8)
                                ).save(os.path.join(save_path, sample['name'][0]))


if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.config)
    test(opt, args)
