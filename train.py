import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import AverageMeter, set_seed
import torch.nn as nn


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model = nn.DataParallel(model)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # Set random seed for this experiment
    set_seed(opt.seed)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        print('Dataset size:', len(dataset))
        meters_trn = {stat: AverageMeter() for stat in model.module.loss_names}
        opt.stage = 'coarse' if epoch < opt.coarse_epoch else 'fine'
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            x, cam2world, cam2azi, masks = model.module.set_input(data)  # unpack data from dataset and apply preprocessing
            output = model(x, cam2world, cam2azi, masks, epoch=epoch, iter=total_iters)  # feedforward and calculate loss
            loss_recon, loss_perc, loss_silhouette, vis = \
                output['loss_recon'], output['loss_perc'], output['loss_silhouette'], output['vis_dict']
            loss = loss_recon.mean() + loss_perc.mean()
            loss += loss_silhouette.mean() if opt.use_occl_silhouette_loss else 0
            layers, avg_grad = model.module.optimize_parameters(loss, opt.display_grad, epoch)   # get gradients, update network weights

            if opt.custom_lr and opt.stage == 'coarse':
                model.module.update_learning_rate()    # update learning rates at the beginning of every step

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                # model.module.compute_visuals()
                # if opt.display_grad:
                #     visualizer.display_grad(layers, avg_grad)

                # display first only
                visualizer.display_current_results({k: v[0] for k, v in vis.items()}, epoch, save_result)

            # losses = model.module.get_current_losses()
            losses = {'recon': loss_recon.mean().item(), 'perc': loss_perc.mean().item()}
            losses['silhouette'] = loss_silhouette.mean().item() if opt.use_occl_silhouette_loss else 0
            for loss_name in model.module.loss_names:
                meters_trn[loss_name].update(float(losses[loss_name]))
                losses[loss_name] = meters_trn[loss_name].avg
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                print('learning rate:', model.module.optimizers[0].param_groups[0]['lr'])

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.module.save_networks(save_suffix)

            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        if not opt.custom_lr:
            model.update_learning_rate()  # update learning rates at the end of every epoch.


