## Training on a custom dataset

### Data organization

There are two components consisting of a training dataset:

1. Multi-view images of training scenes.
2. (Inverted) Extrinsic camera matrix corresponding to each image.

Please organize them in the following format:

```python
image_name = '{}sc{:04d}{}.png'.format(some_string, scene_id, some_other_string)
corresponding_extrinsic_matrix_name = '{}sc{:04d}{}_RT.txt'.format(some_string, scene_id, some_other_string)
```

where ```scene_id``` is the only effective factor for the dataloader to recognize and associate images/extrinsics of a specific scene.
It should range from ```0``` to ```n_scenes-1``` (```n_scenes``` is specified in the running script).
Here we assume no more than 10,000 scenes in the training set by specifying ```"{:04d}"```.
If you have more, simply specify a larger number.
```some_string``` and ```some_other_string``` can be anything that do not include ```"sc{:04d}"```.

An example is:
```
00000_sc0000_img0.png
00000_sc0000_img0_RT.txt
00001_sc0000_img1.png
00001_sc0000_img1_RT.txt
00002_sc0001_img0.png
00002_sc0001_img0_RT.txt
...
```

An (inverted) extrinsic matrix should be a camera-to-world transform
which is composed by camera pose (rotation and translation).
An extrinsic matrix should be something like this:
```
0.87944 -0.30801 0.36292 -4.49662
-0.47601 -0.56906 0.67051 -8.30771
-0.00000 -0.76243 -0.64707 8.01728
0.00000 0.00000 0.00000 1.00000
```
where the uppper-left block is camera orientation matrix and the upper-right vector is the camera location.
Here we assume the origin of world frame is located at the center of the scene,
as we hard-code to put the locality box centered at the world origin 
(it is used to enforce zero density outside the box for foreground slots during early training, 
to help better separate foreground and background).

### Intrinsic parameters

Besides extrinsics, you will also need to specify camera intrinsics in [../models/projection.py](../models/projection.py) by specifying
its ```focal_ratio``` (expressed as focal length divided by number of pixels along X/width axis and Y/height axis).

### Script modification

When you use the training and evaluating scripts in ```/scripts```,
additionally specify ```--fixed_locality```.

According to the scale of your scene (measured by world coordinate units),
you need to modify 
1. the near and far plane (```--near``` and ```--far```) for ray marching
2. the rough scale (does not need to be accurate) of your scene (```--nss_scale```) for normalizing input coordinates.
3. the scale that roughly includes foreground objects (```--obj_scale```) for computing locality box. This should not be too small
such that the box occupied too few pixels (e.g., <80%) when projected to image planes.

### Tips for training on a custom dataset

The background-aware attention brings the advantage of separating background and foreground objects,
allowing cleaner segmentation and scene manipulation.
But it also trades some stability off.
It seems that 
the unsupervised separation of fg/bg somehow makes the module more susceptible to some sort of [rank-collapse problem](https://arxiv.org/abs/2103.03404).
There are two kinds of rank-collapse I have seen during experiments.

- The foreground-background separation is good and the foreground slots are explaining objects, but the foreground slots are few-rank (i.e., some foreground slots decode the same "dim version" of several objects) or even 1-rank.
  
- All the foreground slots decode nothing. In this case you would see zero gradient for all layers of foreground decoder (you can see this from the visdom panel).

In the first case,
adding more training scenes or simply change a random seed is enough.

In the second case, bad learning dynamics leads to zero foreground density even inside the locality boxes early in training, and the foreground decoder is effectively killed. 
Changing random seeds might provide a better initialization and thus bypass the problem,
and you might consider tweaking hyper-parameters related to learning rate scheduler, such as learning rate ```--lr```, its decay speed ```--attn_decay_steps``` and warm-up steps ```--warmup_steps```.
Adding some noise to foreground decoder outputs could also help. 
Hopefully future research can shed light on this problem and solve it once for all.


## Generate your own dataset

We provide an example generation assets and codebase at 
[here](https://office365stanford-my.sharepoint.com/:u:/g/personal/koven_stanford_edu/Ec-vEV0XMxBGpWgx1y6kSkIBOiY_AelngVf2qk2zAHgb_A?e=2gIqGv) 
for the object shape models from ShapeNet and 
[here](https://office365stanford-my.sharepoint.com/:u:/g/personal/koven_stanford_edu/EX0Q_Wmv8EBPqPboJ2QWAqEB18lOQ6aKU2k84tax1YP3Hw?e=lLr16K) 
for the codebase and textures of the Room Diverse dataset.

In ``/image_generation/scripts`` run ``generate_1200shape_50bg.sh`` and then ``render_1200_shapes.sh``.
Don't forget to change the root directory in both scripts.


## Options 
[Class] / {function}

1. Use of certain feature
   1. use_slot_feat
      1. [Encoder] use slot encoder and get object feature
      
   2. use_pixel_feat
      1. [Encoder] use pixel encoder and get pixel feature
      2. you can use this with use_slot_feat option; encoder wrapper will concatenate features.
      3. Warning: now this does not work for batch > 1 per gpu
      
   3. mask_image
      1. [Encoder] mask image before putting it into resnet encoder of pixel encoder
      
   4. mask_image_feature
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
      
   2. use_unoccl_silhouette_loss
      1. [train_model] compute the loss functions of unoccluded silhouette and provided masks; Warning: they are not the same even with GT segmentation and perfectly trained model because GT segmentation can be occluded whereas unoccl_silhouette should be complete object if perfectly trained
      2. [Renderer] {render}: visualize the silhouette using original uorf raw2outputs; it does not consider the transmittance of whole scene for rendering the silhouette; instead, it uses the transmittance of individual k_th object; this silhouette can be used for regularizing the silhouette and removing artifacts during the early training process, but cannot be used until the end if there is any possibility of occlusion.


4. Rendering option
   1. unisurf_render_eq
      1. [Renderer] {render}: change the postprocessing after the decoder output; consider decoder output as alpha, instead of sigma {raw2outputs}


5. Visualization option
   1. visualize_occl_silhouette
      1. [train_model] {compute_visuals}: compute the occluded silhouette
      2. [Renderer] {compute_visual}: compute the occluded silhouette
      
   2. visualize_unoccl_silhouette
      1. [train_model] {compute_visuals}: compute the unoccluded silhouette
      2. [Renderer] {compute_visual}: compute the unoccluded silhouette
      
   3. visualize_weighted_raws: raise NotImplementedError
   
   4. visualize_unweighted_raws: raise NotImplementedError