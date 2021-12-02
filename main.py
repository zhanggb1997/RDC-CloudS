# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/9/8 9:02 
@Author : 弓长广文武
======================================
"""
import argparse
import os
import time
import random
import numpy as np
import torchsummary
import torch
from tensorboardX import SummaryWriter
from torch import nn

from Rep_code.RepCSD import RepCSD
from Rep_code.plot_acc_loss import save_train_test_time, a_l_plot
from Rep_code.CSDNet_v2 import create_CSDNet_V2
from Rep_code.pred import pred_model
from Rep_code.train import train_model
from code_main.evaluate_calculate import calculate_all
from code_net.CDNet_V2.CDNet_V2 import CDnetV2
from code_net.CSDNet.zgb_CSDNet_RES_BN import CSDNet_RES_BN
from code_net.CloudSegNet.zgb_CloudSegNet import CloudSegnet
from code_net.DeeplabV3plus.zgb_Deeplabv3plus_new import deeplabv3plus_new
from code_net.MSCFF.MSCFF import MSCFF
from code_net.RSNet.zgb_RSNet import RSNet
from code_net.UNet.Ran_UNet import U_Net

'''
======================================
@File    :   main.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
# H = ['035', '043']
# for h, hinge in enumerate([0.035, 0.043]):
for i in range(0, 3):

    # =======================  设备检查 ===========================
    # 检查torch版本
    print('===' * 20)
    print('Pytorch 版本 ： ' + torch.__version__)
    print('CUDA    版本 ： ' + torch.version.cuda)
    print('cudnn   版本 ： ' + str(torch.backends.cudnn.version()))
    print('GPU     容量 ： ' + str(torch.cuda.get_device_capability()))
    print('GPU     名称 ： ' + torch.cuda.get_device_name())
    print('===' * 20)

    # # ===============================================================
    # # =========================== 设备设置 ===========================
    # 指定GPU
    CUDA_VISIBLE_DEVICES = 0
    # 判断GPU支持
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 保证可复现性
    torch.backends.cudnn.determinstic = True  # 保证可重复性
    # 随机种子设定
    seed = 0
    torch.manual_seed(seed)  # 固定随机种子
    torch.cuda.manual_seed(seed)  # 固定随机种子
    torch.cuda.manual_seed_all(seed)  # 固定随机种子
    np.random.seed(seed)  # 固定随机种子
    random.seed(seed)  # 固定随机种子
    # # 避免cuDNN的benchmark模式
    torch.backends.cudnn.enabled = True  # 提升计算速率
    torch.backends.cudnn.benchmark = True  # 提升计算速率

    # # ===============================================================
    # # =========================== 超参设置 ===========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--bs", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_channels", type=int, default=3, help="number of channels")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--channels_list", type=list, default=[32, 64, 128, 256, 512], help="number of feature map channels")
    # parser.add_argument("--channels_list", type=list, default=[64, 128, 256, 512, 1024], help="number of feature map channels")
    parser.add_argument("--blocks_list", type=list, default=[1, 1, 1, 1, 1], help="number of feature map channels")
    parser.add_argument("--mul_loss", type=bool, default=True, help="net multi loss is or not")
    parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--init_channel', type=int, default=32, help='initial channel of Encoder')
    parser.add_argument('--beta', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--wtl2', type=float, default=0.998, help='L2 weights in the loss of G')

    # ===============================================================
    # =========================== 路径设置 ===========================
    # data_dir = r'E:\a学生文件\张广斌\data\my_data\CD\M24\rota_en_new'
    data_dir = r'E:\a学生文件\张广斌\data\my_data\CD\S17\no_en'
    # data_dir = r'E:\a学生文件\张广斌\data\other_data\HRC_WHU_Pro\512\no_en'
    # data_dir = r'E:\a学生文件\张广斌\data\my_data\CD\S17_test\no_en'
    # result_dir = r'E:\a学生文件\张广斌\result\21.11.01\CWV_M24'
    result_dir = r'E:\a学生文件\张广斌\result\21.11.01\CWV_S17'
    # result_dir = r'E:\a学生文件\张广斌\result\21.11.01\HRC_WHU'
    parser.add_argument('--trainDir', type=str, default=os.path.join(data_dir, 'train'),
                        help='absolute path of train image')
    parser.add_argument('--valDir', type=str, default=os.path.join(data_dir, 'validation'),
                        help='absolute path of validation image')
    parser.add_argument('--testDir', type=str, default=os.path.join(data_dir, 'test'),
                        help='absolute path of test image')
    parser.add_argument('--outDir', type=str, default=os.path.join(result_dir, 'label_out'),
                        help='absolute path of result output image')
    parser.add_argument('--acloDir', type=str, default=os.path.join(result_dir, 'acc_loss'),
                        help='absolute path of result acc loss image')
    parser.add_argument('--logDir', type=str, default=os.path.join(result_dir, 'log'),
                        help='absolute path of result log')
    parser.add_argument('--witDir', type=str, default=os.path.join(result_dir, 'wits'),
                        help='absolute path of model weights')
    opt = parser.parse_args()
    print(opt)
    print('===' * 20)

    # # ===============================================================
    # # =========================== 模型加载 ===========================
    # # 加载模型
    # model = RSNet(3, 2)
    # model = CloudSegnet(3, 2)
    # model = CSDNet_RES_BN(3, 2)
    # model = MSCFF(3, 2)
    # model = U_Net(3, 1)
    # model = deeplabv3plus_new(3, 1)
    # model = CDnetV2(2)
    # ************************************************************
    # =================== Epoch 80 / 100==================
    # model = create_CSDNet_V2(opt)
    model = RepCSD(num_classes=2, num_channels_list=opt.channels_list, use_atten=False)

    # MODEL_TYPE = 'RepCSD_CA-k5_SA-k7d1_dila12_DS1r15' + '_M24_noe' + str(i) + '_EP_' + str(opt.epochs) + '_' + 'BS=' + str(opt.bs) + '_LR=' + str(opt.lr)
    MODEL_TYPE = 'RepCSD_CA-k5_SA-k7d2_dila33_DS' + '_S13_noe' + str(i) + '_EP_' + str(opt.epochs) + '_' + 'BS=' + str(opt.bs) + '_LR=' + str(opt.lr)
    # MODEL_TYPE = 'RepCSD_CS_SA_avgmax_add_conv_cheng_sig_DS_M24_noen' + str(i) + '_EP_' + str(opt.epochs) + '_' + 'BS=' + str(opt.bs) + '_LR=' + str(opt.lr)
    # MODEL_TYPE = 'RepCSD_ECA_catafter_sparse_dila12_DS_HRC_noen' + str(i) + '_EP_' + str(opt.epochs) + '_' + 'BS=' + str(opt.bs) + '_LR=' + str(opt.lr)
    # MODEL_TYPE = 'RepCSD_down1234up567_64128_CASA_catafter_sparse_dila12_DS_M24_noen' + str(i) + '_EP_' + str(opt.epochs) + '_' + 'BS=' + str(opt.bs) + '_LR=' + str(opt.lr)
    # MODEL_TYPE = 'CloudSegNet_S13_roen_new' + str(i) + '_EP_' + str(opt.epochs) + '_' + 'BS=' + str(opt.bs) + '_LR=' + str(opt.lr)
    # MODEL_TYPE = 'RepCSD_se_up_dila24_1-8_noen' + str(i) + '_epoch_' + str(opt.epochs) + '_' + 'batchsize=' + str(opt.bs) + '_LR=' + str(opt.lr)
    print(MODEL_TYPE)
    print('===' * 20)
    if 'CSD' in MODEL_TYPE:
        opt.mul_loss = True
    else:
        opt.mul_loss = False

    # # # ===============================================================
    # # # =========================== 模型训练 ===========================
    # # # 显示模型图结构
    # with SummaryWriter(log_dir=opt.logDir + '\' + MODEL_TYPE + 'graph', comment=MODEL_TYPE) as w:
    #     w.add_graph(model, torch.rand(opt.bs, opt.num_channels, opt.img_size, opt.img_size))
    # 模型总结
    torchsummary.summary(model.cuda(), (opt.num_channels, opt.img_size, opt.img_size))
    # 放入GPU
    model.to(device)
    # 计时开始训练
    train_time1 = time.time()
    # 训练
    train_history = train_model(model, device, opt, MODEL_TYPE, 0)
    # 计时结束训练
    train_time2 = time.time()

    # ===============================================================
    # =========================== 模型数据 ===========================
    # 保存参数
    torch.save(obj=model.to(device), f=opt.witDir + '/' + MODEL_TYPE + '_last_wts.pth')
    # model = model.eval()
    # torch.save(model.state_dict(), opt.witDir + '/' + MODEL_TYPE + '_last_wts.pth')

    # ===============================================================
    # =========================== 模型预测 ===========================
    # 出图路径
    out_dir = opt.outDir + '/' + MODEL_TYPE
    while os.path.isdir(out_dir):
        out_dir += '_0'
    os.mkdir(out_dir)
    # 测试+制图计时开始
    pred_time1 = time.time()
    # 预测
    pred_model(model, device, opt, out_dir, flag=MODEL_TYPE)
    # 测试计时结束
    pred_time2 = time.time()

    # ===============================================================
    # =========================== 模型损失图、训练时间、参数量 ===========================
    # 显示数据损失度和精确度
    ACC_TRA = train_history[0]
    LOSS_TRA = train_history[1]
    ACC_VAL = train_history[2]
    LOSS_VAL = train_history[3]

    a_l_plot(ACC_TRA, ACC_VAL, LOSS_TRA, LOSS_VAL,
        MODEL_TYPE,
        opt.acloDir,
        ('b', 'Train accuracy'),
        ('r', 'Validation accuracy'),
        ('b', 'Train loss'),
        ('r', 'Validation loss'))

    print(time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒'))

    # 写入训练时间 预测时间 结束时间
    tr_ti_dicts = [
        train_time1,
        train_time2,
        pred_time1,
        pred_time2
    ]
    save_train_test_time(opt.acloDir + '/..', tr_ti_dicts, MODEL_TYPE)

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