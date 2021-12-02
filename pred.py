# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/3/27 10:29 
@Author : 弓长广文武
======================================
"""
import argparse
import os
import random
import time
from glob import glob

import torchsummary
from torchvision import transforms
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np

from Rep_code.RepCSD import RepCSD
from Rep_code.data import PredLoad, PredOut, DatasetLoad
# from model import _netG
from Rep_code.plot_acc_loss import save_test_time
from code_main.evaluate_calculate import calculate_all
from code_net.RSNet.zgb_RSNet import RSNet

'''
======================================
@File    :   pred.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
SNOW = [255, 102, 36]
CLOUD = [66, 97, 255]
BACKGROUND = [214, 217, 212]

COLOR_DICT = np.array([BACKGROUND, CLOUD, SNOW])

def pred_model(model, device, opt, out_dir, flag=''):
    # 数据的加载
    transform = [transforms.ToTensor()]
    # Dataset
    data = PredLoad(opt.testDir, 'image', transforms=transform)
    load = DataLoader(data, opt.bs//4, shuffle=False)
    # 不训练
    # model.train(False)
    model.eval()
    # 放入数据
    with torch.no_grad():
        time_start = time.time()
        pred_result = []
        for image, img_name in tqdm(load, desc='Pred'):
            image = image.to(device=device, dtype=torch.float32)
            if opt.mul_loss:
                pred = model(image)[-1]
            else:
                pred = model(image)
            pred_result.append(pred.detach().cpu().numpy())

        time_end = time.time()
        time_cost = time_end - time_start
        print(flag + 'Pred complete in {:d}m {:.4f}s'.format(int(time_cost // 60), time_cost % 60))

        # 写入训练时间 预测时间 结束时间
        time_dicts = [
            time_start,
            time_end
        ]
        save_test_time(opt.acloDir + '/..', time_dicts, flag)

        for image, img_name in tqdm(load, desc='Pred'):
            image = image.to(device=device, dtype=torch.float32)
            if opt.mul_loss:
                pred = model(image)[-1]
            else:
                pred = model(image)
            pred = pred.detach().cpu().numpy()
            pred_out = PredOut(pred, out_dir, img_name, COLOR_DICT, classes_num=opt.num_classes, flag=flag)
            pred_out.predprocess()

    # # 训练集的最低损失权重拿来用
    # model_tra_best = model.load_state_dict(opt.witDir + '/' + flag + 'best_tra.pth')
    # model_tra_best.eval()
    # # 放入数据
    # with torch.no_grad():
    #     for image, img_name in tqdm(load, desc='Pred'):
    #         image = image.to(device=device, dtype=torch.float32)
    #         if opt.mul_loss:
    #             pred = model_tra_best(image)[-1]
    #         else:
    #             pred = model_tra_best(image)
    #         pred = pred.detach().cpu().numpy()
    #         pred_out = PredOut(pred, out_dir+'_TRA', img_name, COLOR_DICT, classes_num=opt.num_classes, flag=flag)
    #         pred_out.predprocess()


if __name__ == "__main__":
    # =======================  设备检查 ===========================
    print('===' * 20)
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_capability())
    print(torch.cuda.get_device_name())
    print('===' * 20)
    # 指定GPU
    CUDA_VISIBLE_DEVICES = 0
    # 判断GPU支持
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ===============================================================
    # =========================== 固定种子 ===========================
    seed = 0
    torch.manual_seed(seed)  # 固定随机种子
    torch.cuda.manual_seed(seed)  # 固定随机种子
    torch.cuda.manual_seed_all(seed)  # 固定随机种子
    np.random.seed(seed)  # 固定随机种子
    random.seed(seed)  # 固定随机种子
    # ===============================================================
    # =========================== 超参设置 ===========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--bs", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--num_classes", type=int, default=2, help="size of each classify dimension")
    parser.add_argument("--num_channel", type=int, default=3, help="number of image channels")
    parser.add_argument("--mul_loss", type=bool, default=True, help="net multi loss is or not")
    # parser.add_argument("--mul_loss", type=bool, default=False, help="net multi loss is or not")
    parser.add_argument("--channels_list", type=list, default=[32, 64, 128, 256, 512], help="number of feature map channels")
    parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--use_atten', type=bool, default=True, help='is attension module to use')
    # parser.add_argument('--deploy', type=bool, default=True, help='is deploy')
    parser.add_argument('--deploy', type=bool, default=False, help='is deploy')
    # ===============================================================
    # =========================== 命名 =============================
    MODEL_TYPE = 'RepCSD_CA-k5_SA-k7d1_dila12_DS024_S17_noe3_EP_100_BS=8_LR=0.0001_last_'
    # MODEL_TYPE = 'Time-test_cloudseg_noen1_'
    # ===============================================================
    # =========================== 文件夹 =============================
    # parser.add_argument('--testDir', type=str, default=r'E:\a学生文件\张广斌\data\other_data\HRC_WHU_Pro\512\no_en\test',
    #                     help='absolute path of test image')
    parser.add_argument('--testDir', type=str, default=r'E:\a学生文件\张广斌\data\my_data\CD\M24\rota_en_new\test',
                        help='absolute path of test image')
    # parser.add_argument('--testDir', type=str, default=r'E:\a学生文件\张广斌\data\my_data\CD\S17\no_en\test512',
    #                     help='absolute path of test image')
    # result_dir = r'E:\a学生文件\张广斌\result\21.11.01\HRC_WHU'
    result_dir = r'E:\a学生文件\张广斌\result\21.11.01\CWV_M24'
    # result_dir = r'E:\a学生文件\张广斌\result\21.11.01\CWV_S17'
    parser.add_argument('--outDir', type=str, default=os.path.join(result_dir, 'label_out'),
                        help='absolute path of result output image')
    # parser.add_argument('--witDir', type=str, default=r'E:\a学生文件\张广斌\code\Rep_code\DEPLOY_CA-k5_SA-k7d2_dila12_DS_S13_noe1_EP_50_BS=8_LR=0.0001_best.pth',
    #                     help='absolute path of result output image')
    parser.add_argument('--witDir', type=str, default=r'E:\a学生文件\张广斌\result\21.11.01\CWV_M24\wits\RepCSD_CA-k5_SA-k7d1_dila12_DS1r15_M24_noe4_EP_100_BS=8_LR=0.0001_last_wts.pth',
                        help='absolute path of result output image')
    # parser.add_argument('--witDir', type=str, default=r'E:\a学生文件\张广斌\result\21.11.01\HRC_WHU\wits\\'
    #                     + 'RepCSD_CA-cat_sparse-DS_HRC_noen6_EP_80_BS=8_LR=0.0001best_tra.pth',
    #                     help='absolute path of result output image')
    parser.add_argument('--acloDir', type=str, default=os.path.join(result_dir, 'acc_loss'),
                        help='absolute path of result acc loss image')
    opt = parser.parse_args()

    # MODEL_TYPE = 'RepCSD_30311_wits2_Epoch_{:d}_BS_{:d}'.format(opt.epochs, opt.bs)
    # ===============================================================
    # =========================== 模型加载 ===========================
    # 加载模型
    # model = RSNet(3, 2)
    # modelG = _netG(opt)
    model = RepCSD(num_classes=opt.num_classes, num_channels_list=opt.channels_list, use_atten=opt.use_atten, deploy=opt.deploy)
    print('===' * 20)

    # ===============================================================
    # =========================== 加载数据 ===========================
    # 加载参数
    # model.load_state_dict(torch.load(opt.witDir))
    # # 加载模型
    model = torch.load(opt.witDir)
    # # # 重新配置对应的权重
    # # # 模型中的结构字典
    # model_dict = model.state_dict()
    # # 预训练好的权重字典
    # pre_trained_dict = torch.load(opt.witDir)
    # # 重新配置与训练权重，减去不匹配层
    # # pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if (k in model_dict and 'out.o.weight' not in k and 'out.o.bias' not in k)}
    # # 查看pred中的字典是否和model总的字典相匹配
    # for k, v in model_dict.items():
    #     if (k in pre_trained_dict):
    #         print(k + "\t\t\t\t\t\t\t\t : yes")
    #     else:
    #         print(k + "\t\t\t\t\t\t\t\t : no")
    # ###### 将重新配置的训练权重交于model_dict对其进行更新
    # ######### model_dict.update(pre_trained_dict)
    # # 加载权重
    # model.load_state_dict(pre_trained_dict)
    # model = model.eval()
    # ===============================================================
    # =========================== 模型放入 ===========================
    # 模型总结
    torchsummary.summary(model.cuda(), (opt.num_channel, opt.img_size, opt.img_size))
    # 放入GPU
    model.to(device)
    # ===============================================================
    # =========================== 模型预测 ===========================
    # 出图路径
    out_dir = opt.outDir + '/' + MODEL_TYPE
    while os.path.isdir(out_dir):
        out_dir += '_0'
    os.mkdir(out_dir)
    # 预测
    pred_model(model, device, opt, out_dir, flag=MODEL_TYPE)
    # 自动进行评估
    time.sleep(10)
    true_dir = os.path.join(opt.outDir, 'true')
    pre_dir = out_dir

    now_time = time.strftime('%Y%m%d%H%M%S')
    txt_name = os.path.split(pre_dir)[1] + now_time + '.txt'
    txt_dir = os.path.join(opt.outDir, txt_name)
    # (真值图像路径，预测图像路径，保存结果路径，结果标题，需要对比的亮度值(灰度图默认不用填), 是否是灰度图)
    calculate_all(true_dir, pre_dir, txt_dir, MODEL_TYPE + '_预测结果', is_gray=True)
    # calculate_all(true_dir, pre_dir, txt_dir, MODEL_TYPE + '_预测结果', [[66, 97, 255], [255, 102, 36], [214, 217, 212]])
