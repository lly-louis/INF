import numpy as np
from imageio.v3 import imread
from torch import Tensor
from torch.utils.data import Dataset
from einops import rearrange


class RAWsRGBSamplingDataset(Dataset):

    def __init__(self, raw_img_list, srgb_img_list, sample_rate, crop_size=None, 
                 spatial_embed=False, extend_samp_win=True):
        
        self.sample_rate     = sample_rate
        self.raw_img_list    = raw_img_list
        self.srgb_img_list   = srgb_img_list
        self.crop_size       = crop_size
        self.spatial_embed   = spatial_embed
        self.extend_samp_win = extend_samp_win
    

    def read_img(self, raw_image_path, srgb_image_path):
 
        raw_demosaic = imread(raw_image_path) / (2 ** 16 - 1)  
        srgb_img     = imread(srgb_image_path) / (2 ** 16 - 1)

        if (raw_demosaic.shape[0] - raw_demosaic.shape[1]) * (srgb_img.shape[0] - srgb_img.shape[1]) < 0:
            srgb_img = np.rot90(srgb_img, k=1)
        
        spatial = np.dstack(np.meshgrid(np.linspace(0, 1, srgb_img.shape[1]), np.linspace(0, 1, srgb_img.shape[0])))

        if self.crop_size:
            raw_demosaic, srgb_img, spatial = self.img_crop(raw_demosaic, srgb_img, spatial)
        
        raw_sample, srgb_sample, spatial_sample = self.sampling(raw_demosaic, srgb_img, spatial, self.sample_rate)
        
        if self.crop_size and self.extend_samp_win:
            raw_sample, srgb_sample, spatial_sample = self.extend_samples(raw_sample, srgb_sample, spatial_sample)
        
        return (Tensor(raw_demosaic).float(), 
                Tensor(srgb_img).float(),
                Tensor(spatial).float(),
                Tensor(raw_sample).float(),
                Tensor(srgb_sample).float(),
                Tensor(spatial_sample).float())
    

    def img_crop(self, raw_demosaic, srgb_img, spatial):
        '''
        Split the image into patches.
        '''
        p1, p2 = self.crop_size[0], self.crop_size[1]
        raw_demosaic = rearrange(raw_demosaic, '(h p1) (w p2) c -> (h w) p1 p2 c', p1=p1, p2=p2)
        srgb_img = rearrange(srgb_img, '(h p1) (w p2) c -> (h w) p1 p2 c', p1=p1, p2=p2)
        spatial = rearrange(spatial, '(h p1) (w p2) c -> (h w) p1 p2 c', p1=p1, p2=p2)

        return raw_demosaic, srgb_img, spatial


    def sampling(self, raw_img, srgb_img, spatial, sample_rate=0.015):
        '''
        Uniform sampling.
        '''
        shape = raw_img.shape[1:] if self.crop_size else raw_img.shape
        sample_step = int(np.sqrt(1 / sample_rate))  # for example, 0.2% -> an interval of 22
        sample_start = min(shape[0] % sample_step // 2, shape[1] % sample_step // 2)
        
        y_slice, x_slice = (slice(sample_start, (shape[0]-1), sample_step), 
                            slice(sample_start, (shape[1]-1), sample_step))
                
        if self.crop_size:
            return raw_img[:, y_slice, x_slice, :], srgb_img[:, y_slice, x_slice, :], spatial[:, y_slice, x_slice, :]
        else:
            return raw_img[y_slice, x_slice, :], srgb_img[y_slice, x_slice, :], spatial[y_slice, x_slice, :]
    
    
    def extend_samples(self, raw_sample, rgb_sample, spa_sample):
        '''
        Extend the patch to its neighbours for the patch-specific INF.
        '''
        h, w = self.crop_size[2], self.crop_size[3]
        
        new_raw_sample  = np.expand_dims(np.zeros_like(raw_sample), axis=1).repeat(9, 1)
        new_rgb_sample  = np.expand_dims(np.zeros_like(rgb_sample), axis=1).repeat(9, 1)
        new_spa_sample  = np.expand_dims(np.zeros_like(spa_sample), axis=1).repeat(9, 1)

        extend_win = np.array([[-w-1, -w, -w+1],
                               [ -1,   0,   1],
                               [ w-1,  w,  w+1]])
        
        for i in range(raw_sample.shape[0]):
            win = extend_win + i
            
            if i % w == 0:
                win = win[:, 1:]
            if (i+1) % w == 0:
                win = win[:, :-1]
            if 0 <= i < w:
                win = win[1:, :]
            if (h-1)*w <= i < h*w:
                win = win[:-1, :]
            
            for k, num in enumerate(win.flat):
                new_raw_sample[i, k, ...] = raw_sample[num, ...]
                new_rgb_sample[i, k, ...] = rgb_sample[num, ...]
                new_spa_sample[i, k, ...] = spa_sample[num, ...]
            
        return (rearrange(new_raw_sample, 'b (k1 k2) p1 p2 c -> b (k1 p1) (k2 p2) c', k1=3),
                rearrange(new_rgb_sample, 'b (k1 k2) p1 p2 c -> b (k1 p1) (k2 p2) c', k1=3),
                rearrange(new_spa_sample, 'b (k1 k2) p1 p2 c -> b (k1 p1) (k2 p2) c', k1=3))


    def __len__(self):
        return len(self.raw_img_list)


    def __getitem__(self, idx):
        raw_image_path = self.raw_img_list[idx]
        srgb_image_path = self.srgb_img_list[idx]
        file_name = srgb_image_path.rpartition('.')[0].rpartition('/')[2]
        
        raw_img, srgb_img, spatial, raw_sample, srgb_sample, spatial_sample = self.read_img(raw_image_path, srgb_image_path)
        
        return {'raw': raw_img, 'srgb': srgb_img, 'spatial': spatial, 'file_name': file_name,
                'raw_sample': raw_sample, 'srgb_sample': srgb_sample, 'spatial_sample': spatial_sample}
