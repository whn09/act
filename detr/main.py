# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=False)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    return parser


def visualize_model(model):
    # pip install tensorboard onnx netron
    from torch.utils.tensorboard import SummaryWriter
    import torchvision.transforms as transforms
    
    # 创建一个包装类，处理None输入和normalize
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        def forward(self, qpos, image, actions, is_pad):
            # 应用模型中相同的预处理
            image = self.normalize(image)
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            env_state = None
            return self.model(qpos, image, env_state, actions, is_pad)
    
    # 创建每个输入的模拟数据
    batch_size = 1

    # 假设的尺寸，请根据实际情况调整
    image_shape = (batch_size, 1, 3, 480, 640)  # 例如：(批次大小, 通道数, 高度, 宽度)
    qpos_shape = (batch_size, 14)  # 例如：(批次大小, 位置维度)
    action_shape = (batch_size, 400, 14)  # 例如：(批次大小, 动作维度)
    is_pad_shape = (batch_size, 400)  # 例如：(批次大小, 1)

    # 创建dummy tensors
    dummy_qpos_data = torch.randn(*qpos_shape)
    dummy_image_data = torch.randn(*image_shape)
    dummy_action_data = torch.randn(*action_shape)
    dummy_is_pad = torch.zeros(*is_pad_shape)
    
    # 包装模型以便TensorBoard可视化
    model_wrapper = ModelWrapper(model)

    # 对于TensorBoard可视化
    writer = SummaryWriter('logs/model_visualization')
    writer.add_graph(model_wrapper, [dummy_qpos_data, dummy_image_data, dummy_action_data, dummy_is_pad])
    writer.close()

    # 对于ONNX导出
    torch.onnx.export(
        model_wrapper,  # 要导出的模型
        (dummy_qpos_data, dummy_image_data, dummy_action_data, dummy_is_pad),  # 模拟输入的元组
        "model.onnx",  # 输出文件名
        input_names=["qpos_data", "image_data", "action_data", "is_pad"],  # 输入名称
        output_names=["output"],  # 输出名称
        dynamic_axes={  # 动态轴（如果有批次大小是可变的）
            "qpos_data": {0: "batch_size"},
            "image_data": {0: "batch_size"},
            "action_data": {0: "batch_size"},
            "is_pad": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    # netron model.onnx
    

def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    # visualize_model(model)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

