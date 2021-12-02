# -*- coding: utf-8 -*-
"""
======================================
@Create Time : 2021/8/5 8:47 
@Author : 弓长广文武
======================================
"""
import time
import numpy as np
import torch
from torch import nn, optim
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision import datasets

from Rep_code.data import DatasetLoad
from tqdm import tqdm
from colorama import Fore



'''
======================================
@File    :   trian.py    
@Contact :   zhanggb1997@163.com
@Content :
======================================
'''
def train_model(model, device, opt, model_type, hinge=0):
    # 数据的加载
    transform = [transforms.ToTensor()]
    tra_data = DatasetLoad(opt.trainDir, 'image', 'label', classes_num=opt.num_classes, image_mode=1, transforms=transform)
    tra_load = DataLoader(tra_data, batch_size=opt.bs, shuffle=False, pin_memory=True, drop_last=False)
    val_data = DatasetLoad(opt.valDir, 'image', 'label', classes_num=opt.num_classes, image_mode=1, transforms=transform)
    val_load = DataLoader(val_data, batch_size=opt.bs, shuffle=False, pin_memory=True, drop_last=False)

    # 优化器定义
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # # 学习率的调整
    # lr_scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=20, verbose=True)

    # 损失函数定义
    if opt.num_classes > 2:
        criterion = nn.CrossEntropyLoss()  # 多分类
    else:
        criterion = torch.nn.BCEWithLogitsLoss()  # 二分类

    # 初始损失正无穷化
    loss_best_val = float('inf')
    loss_best_tra = float('inf')
    LOSS_VAL, LOSS_TRA, ACC_VAL, ACC_TRA = [], [], [], []

    # 构建可视化损失和精确度图
    writer = SummaryWriter(opt.logDir, comment='Acc_Loss_Show')

    # 清除缓存
    torch.cuda.empty_cache()

    # 开始训练
    start = time.time()
    for epoch in range(opt.epochs):
        print('\n' + '*' * 60)
        print('=================== Epoch {} / {}=================='.format(epoch+1, opt.epochs))

        for phase in ['Train', 'Valid']:
            # 预先定义一个轮次中整体的损失、精度、步数
            run_loss = 0.0
            run_acc = 0.0
            step = 0

            # 判断模式
            if phase == 'Train':
                data_load = tra_load
                model.train(True)  # 调整为训练模式
                # 进度条
                tqdm_loader = tqdm(data_load)  # # 训练迭代器 创建显示进度条的对象
                tqdm_loader.bar_format = '{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET)  # 设置进度条样式属性
                tqdm_loader.unit = 'iterate'  # 每一次迭代为单位计算时间
                # 不断馈送数据
                for image, label in tqdm_loader:
                    step += 1  # 当前批次数
                    tqdm_loader.set_description('epoch:{}-trian:{}'.format(epoch + 1, step))  # 设置tqdm左边显示内容

                    # 优化器梯度清空
                    optimizer.zero_grad()  # 清除梯度值

                    # 数据拷贝到device中
                    image = image.to(device=device, dtype=torch.float32, non_blocking=True)
                    label = label.to(device=device, dtype=torch.float32, non_blocking=True)

                    # # 启用AMP自动混合精度模式
                    # with autocast():
                    pred = model(image)  # 预测
                    if not opt.mul_loss:  # 不是多损失
                        # loss = criterion(pred, label.long())  # 多分类
                        # acc = acc_metric(pred, label.long())  # 多分类
                        loss = criterion(np.squeeze(pred), label)  # 二分类
                        pred_ = nn.Sigmoid()(pred)
                        acc = acc_metric(np.squeeze(pred_), label)  # 二分类
                    else:  # 多损失情况
                        # loss = multi_loss(pred, label.long(), epoch)  # 多分类
                        # acc = multi_acc_metric(pred, label.long())  # 多分类
                        loss = multi_loss(pred, label, epoch, hinge)  # 二分类
                        pred_ = nn.Sigmoid()(pred[-1])
                        acc = multi_acc_metric(pred_, label)  # 二分类

                    loss = loss.requires_grad_()  # 使其能具有梯度属性

                    # 进度条右侧更新损失和精度
                    tqdm_loader.set_postfix(
                        loss_acc='----Loss:{:.5f}----Acc:{:.5f}'.format(loss.item(), acc.item()))

                    # 计算当前epoch中的整体acc\loss
                    run_acc += float(acc) * data_load.batch_size
                    run_loss += float(loss) * data_load.batch_size

                    # # # 更新参数梯度下降反向传播
                    loss.backward()
                    optimizer.step()

            # 验证
            else:
                data_load = val_load
                model.train(False)
                model.eval()
                # 进度条
                tqdm_loader = tqdm(data_load)  # # 验证迭代器  创建显示进度条的对象
                tqdm_loader.bar_format = '{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.RESET)  # 设置进度条样式属性
                tqdm_loader.unit = 'iterate'  # 每一次迭代为单位计算时间

                # 不断馈送数据
                for image, label in tqdm_loader:
                    step += 1  # 当前批次数
                    tqdm_loader.set_description('epoch:{}-trian:{}'.format(epoch + 1, step))  # 设置tqdm左边显示内容

                    # 数据拷贝到device中
                    image = image.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)

                    # # # 启用AMP自动混合精度模式
                    # with autocast():
                    with torch.no_grad():
                        pred = model(image)  # 预测
                        if opt.mul_loss:  # 多输出
                            if opt.num_classes > 2:
                                loss = multi_loss_val(pred, label.long())  # 多分类
                                acc = multi_acc_metric_val(pred, label.long())  # 多分类
                            else:
                                loss = multi_loss_val(pred, label)  # 二分类
                                pred_ = nn.Sigmoid()(pred[-1])
                                acc = multi_acc_metric_val(pred_, label)  # 二分类
                        else:
                            if opt.num_classes > 2:
                                loss = criterion(pred, label.long())  # 多分类
                                acc = acc_metric(pred, label.long())  # 多分类
                            else:
                                loss = criterion(np.squeeze(pred), label)  # 二分类
                                pred_ = nn.Sigmoid()(pred)
                                acc = acc_metric(np.squeeze(pred_), label)  # 二分类

                    # 进度条右侧更新损失和精度
                    tqdm_loader.set_postfix(
                        loss_acc='----Loss:{:.5f}----Acc:{:.5f}'.format(loss.item(), acc.item()))

                    # 计算当前epoch中的整体acc\loss
                    run_acc += float(acc) * data_load.batch_size
                    run_loss += float(loss) * data_load.batch_size

            epoch_loss = run_loss / len(data_load.dataset)
            epoch_acc = run_acc / len(data_load.dataset)

            # print('{} Loss: {:.5f} Acc: {:.5f}'.format(phase, epoch_loss, epoch_acc))
            # 建立损失精度图
            if phase == 'Train':
                LOSS_TRA.append(epoch_loss), ACC_TRA.append(epoch_acc)
                writer.add_scalar('train_loss', epoch_loss, epoch + 1)
                writer.add_scalar('train_acc', epoch_acc, epoch + 1)
                # # 画出层中权值的分布情况
                # for name, param in model.named_parameters():
                #     writer.add_histogram(
                #         name, param.clone().data.numpy(), epoch_index)
                # 监控测试集，保存tra_loss最小的网络参数
                if epoch_loss < loss_best_tra:
                    loss_last = loss_best_tra
                    loss_best_tra = epoch_loss
                    torch.save(model.state_dict(), opt.witDir + '/' + model_type + 'best_tra.pth')
                    print('Train_loss improved from {:.5f} to {:.5f} and save to {}'.format(loss_last, loss_best_tra,
                                                                                            model_type + '_best_tra.pth'))
                else:
                    print('Train_loss didn\'t improved from {:.5f}'.format(loss_best_tra))

            elif phase == 'Valid':
                LOSS_VAL.append(epoch_loss), ACC_VAL.append(epoch_acc)
                writer.add_scalar('val_loss', epoch_loss, epoch + 1)
                writer.add_scalar('val_acc', epoch_acc, epoch + 1)
                # 监控验证集，保存val_loss最小的网络参数
                if epoch_loss < loss_best_val:
                    loss_last = loss_best_val
                    loss_best_val = epoch_loss
                    # torch.save(model.state_dict(), opt.witDir + '/' + model_type + 'best_val.pth')
                    print('Val_loss improved from {:.5f} to {:.5f} and save to {}'.format(loss_last, loss_best_val,
                                                                                          model_type + '_best_val.pth'))
                else:
                    print('Val_loss didn\'t improved from {:.5f}'.format(loss_best_val))
                # if (epoch > 20) & (epoch % 5 == 0):
                #     torch.save(model.state_dict(), opt.witDir + '/' + model_type + '_EP' + str(epoch) + '.pth')

                # num_list = (2, 5, 10, 30, 60)
                # if (epoch + 1) in num_list:
                #     torch.save(obj=model.to(device), f=model_save_path + 'EPOCH_' + str(num_list[temp]) + '_wts.pth')
                #     print('\nSave model on epoch {}'.format(str(num_list[temp])))
                #     temp += 1

        if epoch == 0:
            time_all = 0
        time_epoch = time.time() - time_all - start
        print(
            'Epoch:{}  train_loss:{:.5f}  train_acc:{:.5f}  val_loss:{:.5f}  val_acc:{:.5f}    Training and Validation 1 epoch in {:.0f}m {:.2f}s'
            .format(epoch + 1, LOSS_TRA[epoch], ACC_TRA[epoch], LOSS_VAL[epoch], ACC_VAL[epoch], time_epoch // 60,
                    time_epoch % 60))
        time_all += time_epoch

    # # 保存模型
    # model = model.eval()
    # torch.save(model.state_dict(), opt.witDir + '/' + model_type + '_last_wts3.pth')

    time_elapsed = time.time() - start
    print('Training complete in {:d}m {:.2f}s'.format(int(time_elapsed // 60), time_elapsed % 60))
    writer.close()

    return ACC_TRA, LOSS_TRA, ACC_VAL, LOSS_VAL

def acc_metric(pred, label):
    # return (pred.argmax(dim=1) == label.cuda()).float().mean()  # 多分类
    acc = (abs(label.cuda() - pred)).float().mean()  # 二分类
    return 1 - acc

def multi_acc_metric(pred, label):
    # # acc5 = 0.05 * (pred[0].argmax(dim=1) == label.cuda()).float().mean()
    # # acc1 = 0.1 * (pred[0].argmax(dim=1) == label.cuda()).float().mean()
    # # acc2 = 0.15 * (pred[1].argmax(dim=1) == label.cuda()).float().mean()
    # # acc3 = 0.25 * (pred[2].argmax(dim=1) == label.cuda()).float().mean()
    # acc = (pred[-1].argmax(dim=1) == label.cuda()).float().mean()
    # # return acc1 + acc2 + acc3 + acc
    # return acc
    acc = (abs(label.cuda() - pred[-1])).float().mean()  # 二分类
    return 1 - acc

def multi_acc_metric_val(pred, label):
    # acc1 = (pred[-1].argmax(dim=1) == label.cuda()).float().mean()  # 多分类
    # return acc1
    acc = (abs(label.cuda() - pred[-1])).float().mean()  # 二分类
    return 1 - acc

def multi_loss_val(pred, label):
    # criterion = nn.CrossEntropyLoss()  # 多分类
    # loss1 = criterion(pred[-1], label)

    criterion = torch.nn.BCEWithLogitsLoss()  # 二分类
    loss1 = criterion(np.squeeze(pred[-1]), label)
    return loss1

def multi_loss(pred, label, epoch, hinge):
    # criterion = nn.CrossEntropyLoss()  # 多分类
    # loss = criterion(pred[-1], label)
    # loss0 = criterion(pred[0], label)
    # loss1 = criterion(pred[1], label)
    # loss2 = criterion(pred[2], label)

    criterion = torch.nn.BCEWithLogitsLoss()  # 二分类
    loss = criterion(np.squeeze(pred[-1]), np.squeeze(label))
    loss0 = criterion(np.squeeze(pred[0]), np.squeeze(label))
    loss1 = criterion(np.squeeze(pred[1]), np.squeeze(label))
    loss2 = criterion(np.squeeze(pred[2]), np.squeeze(label))
    # if epoch < 49:
    # if epoch < 30:
    # if epoch < 66:
    #     loss0 = criterion(pred[0], label)
    #     loss1 = criterion(pred[1], label)
    #     loss2 = criterion(pred[2], label)
    #     # loss3 = criterion(pred[3], label)
    #     ratio = 1 / (2 * (1 + (int((epoch + 1) / 10))))
    # ratio = 1 / ( (1 + (int((epoch + 1) / 15))))
    #     # ratio = 1 / (2 * (1 + (int((epoch + 1) / 6))))
    #     # ratio = 1 / (2 + (0.2 * epoch))
    #     # if epoch == 2:
    #     #     i = 1
    #     # print('loss0:{:.5f}   loss1:{:.5f}   loss2::{:.5f}   loss:{:.5f}   lossall:{:.5f}'.format(loss0, loss1, loss2, loss, loss0 * 0.2 + loss1 * 0.3 + loss2 * 0.5))
    # return (loss0 * 0.2 + loss1 * 0.3 + loss2 * 0.5) * ratio + loss * (1 - ratio)
    #     # return (loss0 * 0.1 + loss1 * 0.2 + loss2 * 0.3 + loss3 * 0.4) * ratio + loss * (1 - ratio)

    # # loss3 = 0.2 * criterion(pred[2], label)
    loss_all = loss0 * 0.2 + loss1 * 0.3 + loss2 * 0.5
    # #
    if loss_all > hinge:
        return loss_all * 0.5 + loss * 0.5
    else:
        return loss

    # + 0.25 * criterion(pred[2], label) + 0.15 * criterion(pred[1], label) + 0.1 * criterion(pred[0], label)
    # return loss1

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用  cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    model = UNet(3, 3)
    # 将网络拷贝到deivce中
    model.to(device=device)
    # 指定训练集地址，开始训练
    data_path = r'E:\a学生文件\张广斌\data\my_data\CSD_S5\512\last_5000\rota_en\train'
    train_model(model, device, data_path, 10, 1)