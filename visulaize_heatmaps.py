import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.utils.data
from utils.data import get_dataset
from utils.dataset_processing.grasp import detect_grasps,GraspRectangles
from models.common import post_process_output
import cv2
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
matplotlib.use("TkAgg")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str,
                        default="output/epoch_04_iou_0.98",
                        help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default="jacquard", help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, default="E:\\Dataset\\cornell\\cornell grasp data\\09", help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=0, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--vis', type=bool, default=False, help='vis')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=200, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=32, help='Validation Batches')
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args.network)
    print(args.use_rgb,args.use_depth)
    net = torch.load(args.network)
    # net_ggcnn = torch.load('./output/models/211112_1458_/epoch_30_iou_0.75')
    device = torch.device("cuda:0")
    Dataset = get_dataset(args.dataset)

    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=False,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )
    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }
    ld = len(val_data)
    save_directory = "D:/deform/jacquard1"  # 请确保这个目录存在或者代码会创建这个目录
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    with torch.no_grad():
        batch_idx = 0

        for id, (x, y, didx, rot, zoom_factor) in enumerate(val_data):

            print(id)
            print(x.shape)
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']
            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            gs_1 = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=6)
            rgb_img = val_dataset.get_rgb(didx, rot, zoom_factor, normalise=False)

            # fig_dpi = 100
            # fig_size = (9, 7)
            # 保存RGB图像

            rgb_img_path = os.path.join(save_directory, f'{id}_1.png')
            fig, ax = plt.subplots()
            ax.imshow(rgb_img)
            ax.set_axis_off()
            plt.savefig(rgb_img_path, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(9, 9), dpi=100)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(rgb_img)
            ax.set_xlim([0, rgb_img.shape[1]])
            ax.set_ylim([rgb_img.shape[0], 0])
            ax.axis('off')
            for g in gs_1:
                g.plot(ax)

            # plt.show()  # 显示当前图像
            # plt.close(fig)  # 关闭当前图像，开始下一个循环
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            img_save_path = os.path.join(save_directory, f'{id}_0.png')
            plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0)  # 保存图像到指定目录

            # q_img_path = os.path.join(save_directory, f'{id}_2.png')
            # # fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
            # im = ax.imshow(q_out, cmap='jet', vmin=0, vmax=1, aspect='equal')
            # # plt.colorbar(im, ax=ax)
            # ax.set_axis_off()
            # plt.savefig(q_img_path, bbox_inches='tight')
            # plt.close(fig)
            #
            # # 保存角度图像
            # ang_img_path = os.path.join(save_directory, f'{id}_3.png')
            # # fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
            # im = ax.imshow(ang_out, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2, aspect='equal')
            # # plt.colorbar(im, ax=ax)
            # ax.set_axis_off()
            # plt.savefig(ang_img_path, bbox_inches='tight')
            # plt.close(fig)
            #
            # # 保存宽度图像
            # w_img_path = os.path.join(save_directory, f'{id}_4.png')
            # # fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
            # im = ax.imshow(w_out, cmap='jet', vmin=0, vmax=150, aspect='equal')
            # # plt.colorbar(im, ax=ax)
            # ax.set_axis_off()
            # plt.savefig(w_img_path, bbox_inches='tight')
            # plt.close(fig)

            q_img_path = os.path.join(save_directory, f'{id}_2.png')
            fig, ax = plt.subplots()
            im = ax.imshow(q_out, cmap='jet', vmin=0, vmax=1)
            # plt.colorbar(im, ax=ax)
            ax.set_axis_off()
            plt.savefig(q_img_path, bbox_inches='tight')
            plt.close(fig)

            # 保存角度图像
            ang_img_path = os.path.join(save_directory, f'{id}_3.png')
            fig, ax = plt.subplots()
            im = ax.imshow(ang_out, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
            # plt.colorbar(im, ax=ax)
            ax.set_axis_off()
            plt.savefig(ang_img_path, bbox_inches='tight')
            plt.close(fig)

            # 保存宽度图像
            w_img_path = os.path.join(save_directory, f'{id}_4.png')
            fig, ax = plt.subplots()
            im = ax.imshow(w_out, cmap='jet', vmin=0, vmax=150)
            # plt.colorbar(im, ax=ax)
            ax.set_axis_off()
            plt.savefig(w_img_path, bbox_inches='tight')
            plt.close(fig)

