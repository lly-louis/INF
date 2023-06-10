import os
import yaml
import logging
import argparse
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from einops import rearrange
from skimage.metrics import structural_similarity as ssim

from dataset.Dataset_CAM import RAWsRGBSamplingDataset
from model.INF import INF

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

with open("INF_NUS.yaml", 'r', encoding='utf-8') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=str)
args.update(vars(parser.parse_args()))

logging.basicConfig(level=logging.INFO,
                    filename=f"{args['result_path']}/{args['camera']}.log", 
                    filemode='a',
                    format='%(asctime)s [%(levelname)s] %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

#-------------------- prepare dataset --------------------

raw_img_list = sorted(glob(f"{args['data_path']}/{args['camera']}/demosaic/*"))
srgb_img_list = sorted(glob(f"{args['data_path']}/{args['camera']}/sRGB/*"))

win_size = args['NUS_Dataset_size'][args['camera']]
crop_shape = (win_size[0] // win_size[2], win_size[1] // win_size[3])
dataset = RAWsRGBSamplingDataset(raw_img_list, srgb_img_list, 
                                 args['sample_rate'], 
                                 crop_size=(win_size[2], win_size[3], crop_shape[0], crop_shape[1]), 
                                 spatial_embed=True, extend_samp_win=True)        
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

#-------------------- core function --------------------

def reconstruct(args, train_data, test_data):
    network_size = (args['num_layers'], args['hidden_dim'], args['add_layer'])
    model = INF(*network_size, weight_decay=args['weight_decay'])
    model.cuda()

    optim = torch.optim.Adam(model.params, lr=args['lr'])
    scheduler = lr_scheduler.StepLR(optim, step_size=args['scheduler']['step_size'], gamma=args['scheduler']['gamma'])
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(args['epoch']):
        
        optim.zero_grad()

        t_o = model(train_data[0], train_data[1])
        t_loss = loss_fn(t_o, train_data[-1])

        t_loss.backward()
        optim.step()
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        v_o = model(test_data[0], test_data[1])
        v_loss = loss_fn(v_o, test_data[-1])
        v_psnrs = -10 * torch.log10(v_loss).item()
    
    return v_psnrs, v_o

#-------------------- run on the whole dataset --------------------

for sample_batched in tqdm(dataloader):
    
    raw_img  = sample_batched['raw'].cuda()
    result_img  = torch.zeros_like(raw_img)
    raw_img_numpy = raw_img.squeeze(0).detach().cpu().numpy()

    raw_img     = raw_img.squeeze(0).split(1)
    raw_sample  = sample_batched['raw_sample'].cuda().squeeze(0).split(1)
    srgb_sample = sample_batched['srgb_sample'].cuda().squeeze(0).split(1)
    spa_sample  = sample_batched['spatial_sample'].cuda().squeeze(0).split(1)
    srgb_img    = sample_batched['srgb'].cuda().squeeze(0).split(1)
    spatial     = sample_batched['spatial'].cuda().squeeze(0).split(1)

    img_psnr = 0
    for i, srgb in enumerate(srgb_img):
        psnr, result_img[0, i, ...] = reconstruct(args,
                                                  train_data=(srgb_sample[i], spa_sample[i], raw_sample[i]), 
                                                  test_data=(srgb, spatial[i], raw_img[i]))
        img_psnr += psnr
        
    result_img_numpy = result_img.squeeze(0).detach().cpu().numpy()
    recon_raw = rearrange(result_img_numpy, '(h w) p1 p2 c -> (h p1) (w p2) c', h=crop_shape[0], w=crop_shape[1])
    recon_raw[recon_raw<0] = 0
    raw = rearrange(raw_img_numpy, '(h w) p1 p2 c -> (h p1) (w p2) c', h=crop_shape[0], w=crop_shape[1])
    
    img_ssim = ssim(raw, recon_raw, channel_axis=-1)
    logging.info(f"{sample_batched['file_name']} : psnr = {img_psnr / (i+1):4f} ; ssim = {img_ssim:4f}")