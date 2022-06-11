
## Options 
[Class] / {function}
The classes that initialize the classes are often omitted (e.g. End2end initialize the encoder, so need an option, but omitted here, whereas train_model is not.)

1. Change the conditioning the input of NeRF decoder
   1. use_slot_feat
      1. [Encoder] use slot encoder and get object feature
      
   2. use_pixel_feat
      1. [Encoder] use pixel encoder and get pixel feature
      2. you can use this with use_slot_feat option; encoder wrapper will concatenate features.
      3. Warning: now this does not work for batch > 1 per gpu

   3. use_ray_dir_world
      1. [Renderer] use the ray direction in the world (=(x, y, z) unit vector )
      2. There is no meaning of using ray dir in cam coordinates
      3. If you assume diffusive materials, ray_dir might not be necessary
      
   4. mask_image
      1. [Encoder] mask image before putting it into resnet encoder of pixel encoder
      
   5. mask_image_feature
      1. [Encoder] mask image before putting it into resnet encoder of pixel encoder


2. Load mask
   1. gt_seg 
      1. [Dataloader] load segmentation map
      2. [Encoder] masking based on loaded seg
      3. provide mask for spatial sampling
      
   2. use_eisen_seg
      1. [Dataloader] use EISEN segmentation instead of GT segmentation


3. Use of certain loss
   1. use_occl_silhouette_loss
      1. [train_model] compute the loss functions of occluded silhouette and provided masks
      2. [Renderer] render: visualize the silhouette using the rendering equation; it consider the transmittance of whole scene for rendering the silhouette, and this silhouette can be used for silhouette loss with our 2d segmentation map
      3. Occluded silhouette can only be computed when you call renderer.render to render the entire scene (you cannot compute this with individual k_th object because you do not have a transmittance of entire scene)
      4. You can get occluded_silhouette from renderer.render
      5. use_unoccl_silhouette_loss
         1. raise NotImplementedError
         2. [train_model] compute the loss functions of unoccluded silhouette and provided masks; Warning: they are not the same even with GT segmentation and perfectly trained model because GT segmentation can be occluded whereas unoccl_silhouette should be complete object if perfectly trained
         3. [Renderer] {render}: visualize the silhouette using original uorf raw2outputs; it does not consider the transmittance of whole scene for rendering the silhouette; instead, it uses the transmittance of individual k_th object; this silhouette can be used for regularizing the silhouette and removing artifacts during the early training process, but cannot be used until the end if there is any possibility of occlusion.
         4. Unoccluded silhouette can be computed individually with k_th object
         5. You can get unoccluded silhouette from either renderer.render (all objects) or renderer.compute_visual (individual objects)


4. Rendering option
   1. unisurf_render_eq
      1. [Renderer] {render}: change the postprocessing after the decoder output; consider decoder output as alpha, instead of sigma {raw2outputs}
      2. This principle is not right. TODO: think about the distance random sampling. also, our frustum does not have equal average distance; it changes as the ray dir changes (think about how the cam frustum look like in the world).
      3. Also, this method does not work if you have a transparent object and change the distance between the sampling points; the more you sample, the more it become non-transparent because you will add more points inside the objects and alpha will be added more.


5. Visualization option
   1. visualize_occl_silhouette
      1. assert opt.use_occl_silhouette_loss
      2. [train_model] {forward}: compute the occluded silhouette
      3. [Renderer] {render}: compute the occluded silhouette
      
   2. visualize_unoccl_silhouette
      1. [train_model] {compute_visuals}: compute the unoccluded silhouette
      2. [Renderer] {compute_visual}: compute the unoccluded silhouette
      
   3. visualize_mask
      1. [train_model] visualize the loaded mask
   
   4. visualize_attn
      1. raise NotImplementedError



# Unsupervised Discovery of Object Radiance Fields
by [Hong-Xing Yu](https://kovenyu.com), [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/) and [Jiajun Wu](https://jiajunwu.com/) from Stanford University.

![teaser](teaser.gif)

arXiv link: [https://arxiv.org/abs/2107.07905](https://arxiv.org/abs/2107.07905)

Project website: [https://kovenyu.com/uorf](https://kovenyu.com/uorf)



## Environment
We recommend using Conda:
```sh
conda env create -f environment.yml
conda activate uorf-3090
```
or install the packages listed therein. Please make sure you have NVIDIA drivers supporting CUDA 11.0, or modify the version specifictions in `environment.yml`.

## Data and model
Please download datasets and models [here](https://office365stanford-my.sharepoint.com/:f:/g/personal/koven_stanford_edu/Et9SOVcOxOdHilaqfq4Y3PsBsiPGW6NGdbMd2i3tRSB5Dg?e=WRrXIh).
If you want to train on your own dataset or generate your own dataset similar to our used ones, please refer to this [README](data/README.md).

## Evaluation
We assume you have a GPU.
If you have already downloaded and unzipped the datasets and models into the root directory,
simply run
```sh
bash scripts/eval_nvs_seg_chair.sh
```
from the root directory. Replace the script filename with `eval_nvs_seg_clevr.sh`, `eval_nvs_seg_diverse.sh`,
and `eval_scene_manip.sh` for different evaluations. Results will be saved into `./results/`.
During evaluation, the results on-the-fly will also be sent to visdom in a nicer form, which can be accessed from
[localhost:8077](http://localhost:8077).

## Training
We assume you have a GPU with no less than 24GB memory (evaluation does not require this as rendering can be done ray-wise but some losses are defined on the image space),
e.g., 3090. Then run
```shell
bash scripts/train_clevr_567.sh
```
or other training scripts. If you unzip datasets on some other place, add the location as the first parameter:
```shell
bash scripts/train_clevr_567.sh PATH_TO_DATASET
```
Training takes ~6 days on a 3090 for CLEVR-567 and Room-Chair, and ~9 days for Room-Diverse.
It can take even longer for less powerful GPUs (e.g., ~10 days on a titan RTX for CLEVR-567 and Room-Chair).
During training, visualization will be sent to [localhost:8077](http://localhost:8077).

## Bibtex
```
@inproceedings{yu2022unsupervised
  author    = {Yu, Hong-Xing and Guibas, Leonidas J. and Wu, Jiajun},
  title     = {Unsupervised Discovery of Object Radiance Fields},
  booktitle = {International Conference on Learning Representations},
  year      = {2022},
}
```

## Acknowledgement
Our code framework is adapted from Jun-Yan Zhu's [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Some code related to adversarial loss is adapted from [a pytorch implementation of StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).
Some snippets are adapted from pytorch [slot attention](https://github.com/lucidrains/slot-attention) and [NeRF](https://github.com/yenchenlin/nerf-pytorch).
If you find any problem please don't hesitate to email me at koven@stanford.edu or open an issue.
