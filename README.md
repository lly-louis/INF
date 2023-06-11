# Metadata-Based RAW Reconstruction via Implicit Neural Functions <a href="https://colab.research.google.com/drive/1KFNFQgLcQ7HwIFn7fMFh4DiwZ0tb4hyv?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
### [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Metadata-Based_RAW_Reconstruction_via_Implicit_Neural_Functions_CVPR_2023_paper.pdf)]  [[Technical Video](https://www.youtube.com/watch?v=vNLRd-iy1X0)]  [[Colab Demo](https://colab.research.google.com/drive/1KFNFQgLcQ7HwIFn7fMFh4DiwZ0tb4hyv?usp=sharing)]

**Metadata-Based RAW Reconstruction via Implicit Neural Functions**
<br>_Leyi Li, Huijie Qiao, Qi Ye, Qinmin Yang_<br>
In CVPR 2023

## Environment
This code is developed with Python 3.9 and Pytorch 1.13. Make sure that [PyTorch](https://pytorch.org/get-started/locally/) is correctly installed. Install other dependencies via
```bash
pip install numpy einops PyYAML scikit-image imageio opencv-python tqdm
```
## Datasets
We test our method on [NUS dataset](https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html) and [CAM dataset](https://github.com/SamsungLabs/content-aware-metadata).

For NUS dataset, we follow the [previous work](https://openaccess.thecvf.com/content/WACV2021/papers/Punnappurath_Spatially_Aware_Metadata_for_Raw_Reconstruction_WACV_2021_paper.pdf) to use a [software ISP platform](https://karaimer.github.io/camera-pipeline/) to produce demosaiced RAW images and sRGB images. The final file structure is:
```
NUS_dataset
├── OlympusEPL6
│   ├── demosaic
│   │   ├── OlympusEPL6_0001.tif
│   │   ├── ...
│   ├── sRGB
│   │   ├── OlympusEPL6_0001.tif
│   │   ├── ...
├── SamsungNX2000
└── SonyA57
```
For CAM dataset, we remain their original file structure.
## Usage
Modify the `data_path` and `result_path` in the config files (`INF_NUS.yaml` or `INF_CAM.yaml`). Then run
```bash
# For NUS dataset
python test_NUS.py --camera {CAMERA}
# For CAM dataset
python test_CAM.py --camera {CAMERA}
```
where `CAMERA` can be one of [`OlympusEPL6`, `SamsungNX2000`, `SonyA57`]. The results are logged in `{result_path}/{CAMERA}.log`.

You may wonder where are the pretrained models? Actually our method is a completely self-supervised method, which requires no pretraining on any dataset.😆
## Citation
If you find this repository useful for your research, please consider to cite it as
```latex
@inproceedings{li2023metadata,
  title={Metadata-Based RAW Reconstruction via Implicit Neural Functions},
  author={Li, Leyi and Qiao, Huijie and Ye, Qi and Yang, Qinmin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={18196--18205},
  year={2023}
}
```
## Acknowledgement
This implementation is based on / inspired by:
- https://github.com/GlassyWing/fourier-feature-networks (fourier-feature-networks)
- https://github.com/prs-eth/PixTransform (PixTransform)
## Contact
If you have any question about this work, feel free to contact [lileyi@zju.edu.cn](lileyi@zju.edu.cn).