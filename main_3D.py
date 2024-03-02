from dataset import  build_dataset_3D_pretrained,build_dataLoader_3D_pretrained
from tools import wandb_init,setup_default_logging
from dataset import build_dataset, build_dataLoader
from model import build_swin,  build_pretrained, build_resnetDecoder, build_convnext, build_memoryBank, build_Decoder, build_proEncoder, MSFF,Main_model
from model import Swin_promte, internal, Swin_decoder,MPNet
from train import pretrained_train, train_step
import os
from config import cfg
import logging
import torch
import datetime
from config import  dataset3D_cfg
import argparse


dataset_cfg = cfg['dataset']

train_cfg_main = cfg['train']
pretrained_cfg = cfg['train']['pretrain']
train_cfg = cfg['train']['train']

_logger = logging.getLogger('train')

setup_default_logging()
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0")
#device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
_logger.info('Device: {}'.format(device))


def pretrain(target = None):
    
    dataset3D_cfg["target"] = target
    current_time = datetime.datetime.now()

    dataset = build_dataset_3D_pretrained(**dataset3D_cfg)
    dataLoader = build_dataLoader_3D_pretrained(dataset)
    convnext_backbone = build_convnext(device, "./checkpoints/convnext_base_1k_224.pth")
    resnet_decoder = build_resnetDecoder(device)

    pretrained_model = build_pretrained(convnext_backbone, resnet_decoder, device)

    time_str = ""+ str(current_time.month) + "_"+ str(current_time.day)+ "_"+ str(current_time.hour)
    trace_dir = "./tracing/pretrained_"+ time_str + ".csv"

    num_step = pretrained_cfg['num_step']
    
    pretrained_train(
        model = pretrained_model,
        trainloader = dataLoader,
        device=device,
        num_training_steps=num_step,

        savedir=trace_dir,
    )
    torch.save(convnext_backbone.state_dict(),"./checkpoints/backbone/3D_"+target+".pt")


parser = argparse.ArgumentParser(description='add settings')
parser.add_argument('-t','--target', type=str,
                    choices = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope","tire"]
                    , help='add the target')

if __name__ == "__main__":
    args = parser.parse_args()
    target = args.target
    pretrain(target= target)