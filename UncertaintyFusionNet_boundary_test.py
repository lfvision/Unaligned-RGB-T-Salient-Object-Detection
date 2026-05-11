import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.UncertaintyFusionSOD import UncertaintyFusionNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/home/user/huxh/TriTransNet/test_datasets/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
# if opt.gpu_id=='0':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     print('USE GPU 0')
# elif opt.gpu_id=='1':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     print('USE GPU 1')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#load the model
model = UncertaintyFusionNet()
model.load_state_dict(torch.load('./Result_ALL/UncertaintyFusionNet_epoch_best.pth'), strict=True)
model.cuda() 
model.eval()
save_result_path_name = 'boundary_20251023'
# test_datasets = ['UVT20K', 'UVT2000', 'U-VT5000', 'U-VT1000', 'U-VT821']
# test_datasets = ['U-VT5000']
test_datasets = ['VT5000', 'VT1000', 'VT821']
# test_datasets = ['UVT20K', 'UVT2000']
# test_datasets = ['UVT20K']


for dataset in test_datasets:
    save_path = './test_maps/'+save_result_path_name+'/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if "UVT20K" == dataset:
        image_root = 'SOD/UVT20K/Test/RGB/'
        gt_root = 'SOD/UVT20K/Test/GT/'
        depth_root = 'SOD/UVT20K/Test/T/'
    elif "UVT2000" == dataset:
        image_root = 'SOD/UVT2000/RGB/'
        gt_root = 'SOD/UVT2000/GT/'
        depth_root = 'SOD/UVT2000/T/'
    elif "U-VT5000" == dataset:
        image_root = 'SOD/WeaklyAligned/VT5000-Test_unalign/RGB/'
        gt_root = 'SOD/WeaklyAligned/VT5000-Test_unalign/GT/'
        depth_root = 'SOD/WeaklyAligned/VT5000-Test_unalign/T/'
    elif "U-VT1000" == dataset:
        image_root = 'SOD/WeaklyAligned/VT1000_unalign/RGB/'
        gt_root = 'SOD/WeaklyAligned/VT1000_unalign/GT/'
        depth_root = 'SOD/WeaklyAligned/VT1000_unalign/T/'
    elif "U-VT821" == dataset:
        image_root = 'SOD/WeaklyAligned/VT821_unalign/RGB/'
        gt_root = 'SOD/WeaklyAligned/VT821_unalign/GT/'
        depth_root = 'SOD/WeaklyAligned/VT821_unalign/T/'
    elif "VT5000" == dataset:
        image_root = 'SOD/RGBTSOD/VT5000/Test/RGB/'
        gt_root = 'SOD/RGBTSOD/VT5000/Test/GT/'
        depth_root = 'SOD/RGBTSOD/VT5000/Test/T/'
    elif "VT821" == dataset:
        image_root = 'SOD/RGBTSOD/VT821/RGB/'
        gt_root = 'SOD/RGBTSOD/VT821/GT/'
        depth_root = 'SOD/RGBTSOD/VT821/T/'
    elif "VT1000" == dataset:
        image_root = 'SOD/RGBTSOD/VT1000/RGB/'
        gt_root = 'SOD/RGBTSOD/VT1000/GT/'
        depth_root = 'SOD/RGBTSOD/VT1000/T/'
    else:
        print('No dataset named {}'.format(dataset))
        raise NotImplementedError

    
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    print('Testing on {} dataset...'.format(dataset), (test_loader.size))
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth = depth.repeat(1,3,1,1).cuda()
        res, res2, res3, res4,_,_ , b1,b2,b3,b4 = model(image,depth)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res*255)
        # exit()
    print('Test Done!')
