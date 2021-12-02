# -*- coding: utf-8 -*-
'''
======================================
@File    :   plot_acc_loss.py    
@Contact :   zhanggb1997@163.com
@License :   free
======================================
'''
import os
import time

"""
======================================
@Create Time : 2020/10/3 18:59 
--------------------------------------
@Author : ZhangGB 
======================================
"""
import matplotlib.pyplot as plt

def a_l_plot(acc, val_acc, loss, val_loss, modelType, save_path, *args):
    ''':argument'''
    epochs = range(1, len(acc) + 1)

    plt.figure(num=0, figsize=(15, 10))
    plt.plot(epochs, acc, args[0][0], label=args[0][1])
    plt.plot(epochs, val_acc, args[1][0], label=args[1][1])
    plt.title(args[0][1] + args[1][1])
    plt.legend()
    now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒')
    # now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='n', m='y', d='r', h='s', M='f', s='m')
    img_path = os.path.join(save_path, 'acc_' + modelType + now_time)
    img_path = img_path + '.tif'
    plt.savefig(img_path)


    plt.figure(num=1, figsize=(15, 10))
    plt.plot(epochs, loss, args[2][0], label=args[2][1])
    plt.plot(epochs, val_loss, args[3][0], label=args[3][1])
    plt.title(args[2][1] + args[3][1])
    plt.legend()
    now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒')
    img_path = os.path.join(save_path, 'loss_' + modelType + now_time)
    img_path = img_path + '.tif'
    plt.savefig(img_path)


def save_test_acc_loss(dir, dicts, note):
    default_dir = os.path.join(dir, 'all_result.txt')
    file_handle = open(default_dir, mode='a', encoding='utf-8')
    file_handle.write('*******--- ' + note + ' ---*******' + '\n')
    for res_items in dicts:
        file_handle.write(res_items + ': ' + '{:.4f}'.format(dicts.get(res_items)) + ' || \n')
    now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒')
    file_handle.write('####' + now_time + '\n')
    file_handle.writelines('\n')
    file_handle.flush()
    file_handle.close()

def save_train_test_time(dir, time_dicts, note):
    default_dir = os.path.join(dir, 'train_test_time.txt')
    file_handle = open(default_dir, mode='a', encoding='utf-8')
    file_handle.write('*******--- ' + note + ' ---*******' + '\n')
    file_handle.write('####' + '训练时长:' + str(int((time_dicts[1]-time_dicts[0])/3600)) + '时' + str(int(((time_dicts[1]-time_dicts[0])%3600)/60)) + '分'
                      + '%.4f'%(((time_dicts[1]-time_dicts[0])%3600)%60) + '秒' + '\n')
    file_handle.write('####' + '预测时长: %.5f'%(time_dicts[3]-time_dicts[2]) + 's' + '\n')
    now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒')
    file_handle.write('####' + now_time + '\n')
    file_handle.writelines('\n')
    file_handle.flush()
    file_handle.close()

def save_test_time(dir, time_dicts, note):
    default_dir = os.path.join(dir, 'test_time.txt')
    file_handle = open(default_dir, mode='a', encoding='utf-8')
    file_handle.write('*******--- ' + note + ' ---*******' + '\n')
    file_handle.write('####' + '预测时长: %.5f'%(time_dicts[1]-time_dicts[0]) + 's' + '\n')
    now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒')
    file_handle.write('####' + now_time + '\n')
    file_handle.writelines('\n')
    file_handle.flush()
    file_handle.close()