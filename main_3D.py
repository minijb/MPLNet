from dataset import  build_dataset_3D_pretrained,build_dataLoader_3D_pretrained, build_dataset3D, build_dataLoader3D
from tools import wandb_init,setup_default_logging
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
use_wandb = train_cfg_main['use_wandb']

def pretrain(target = None):
    
    dataset3D_cfg["target"] = target
    current_time = datetime.datetime.now()

    dataset = build_dataset_3D_pretrained(**dataset3D_cfg)
    dataLoader = build_dataLoader_3D_pretrained(dataset)
    convnext_backbone = build_convnext(device, "./checkpoints/convnext_base_1k_224.pth", need_del=True)
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




def train_3D(item = None):
    
    if item is None :
        exit()
    
    current_time = datetime.datetime.now()
    tail = str(current_time.day)+"_"+str(current_time.hour)+"_"+str(current_time.second)
    
    if use_wandb:
        wandb_init("3D", "train_"+item+tail, train_cfg_main['train'])
    
    
    dataset_cfg['target'] = item
    backbone_path = os.path.join("./checkpoints/backbone", "3D_"+item+".pt")
    
    
    
    # build dataset ------------------------------------------
    trian_dataset = build_dataset3D(
        **dataset3D_cfg,
        train=True
    )
    
    memory_dataset = build_dataset3D(
        **dataset3D_cfg,
        train=True,
        to_memory=True
    )
    
    test_dataset = build_dataset3D(
        **dataset3D_cfg,
        train=False
    )
    
    trainloader = build_dataLoader3D(
        dataset=trian_dataset,
        train=True
    )
    
    testloader = build_dataLoader3D(
        dataset= test_dataset,
        train=False
    )
    
    # build model ------------------------------------------
    encoder_conv = build_convnext(device, backbone_path)
    
    for param in encoder_conv.named_parameters():
        param[1].requires_grad = False
    
    # encoder_conv = build_convnext(device, "./checkpoints/convnext_base_1k_224.pth")
    
    memoryBank = build_memoryBank(device, memory_dataset, 30)
    memoryBank.update(encoder_conv)
   
    channel_list = [128, 256, 512, 1024]
    ss_list = [64, 32, 16, 8]
    promte_mode = Swin_promte(channel_list, ss_list)
    promte_mode.to(device)
    
    internal_model = internal(channel_list[0:-1])
    internal_model.to(device)
    
    decoder = Swin_decoder()
    decoder.to(device)
    
    main_model = MPNet(encoder_conv, promte_mode, memoryBank, internal_model, decoder)
    main_model.to(device)
    
    for params in main_model.named_parameters():
        if param[0].find("encoder") != -1:
            param[1].requires_grad = False
            
    
    train_step(
        model=main_model,
        dataloader=trainloader,
        validloader=testloader,
        num_training_steps=train_cfg['num_step'],
        log_interval=1,
        eval_interval= 200,
        device=device,
        use_wandb=use_wandb,
        savedir="./save/"+dataset_cfg['target']
    )
    wandb.finish()





parser = argparse.ArgumentParser(description='add settings')
parser.add_argument('-t','--target', type=str,
                    choices = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope","tire"]
                    , help='add the target')


# the path of 3D backbone : 3D_{item}.pt
if __name__ == "__main__":
    args = parser.parse_args()
    if args.target is not None:
        target = args.target
    else:
        print("should input target\n")
        exit()
    train_3D(item= target)