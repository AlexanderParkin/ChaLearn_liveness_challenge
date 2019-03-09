import argparse, os, json
import torch
import torchvision as tv
from utils import transforms

def get_opts():
    opt = argparse.Namespace()
    
    opt.task_name = ''
    opt.exp_name = 'exp1_2stagev2_stage2_seed2'
    opt.fold = 1
    opt.data_root = '/ssd/a.parkin/CASIA-SURF/'
    opt.data_list = 'data/lists/folds_by_fakes/fold{fold_n}/'.format(fold_n=opt.fold)
    
    opt.out_root = 'data/opts/'
    opt.out_path = os.path.join(opt.out_root,opt.exp_name,'fold{fold_n}'.format(fold_n=opt.fold))
    
    ### Dataloader options ###
    opt.nthreads = 32
    opt.batch_size = 128 #280
    opt.ngpu = 4

    ### Learning ###
    opt.freeze_epoch = 0
    opt.optimizer_name = 'Adam'
    opt.weight_decay = 0
    opt.lr = 2e-5
    opt.lr_decay_lvl = 0.5
    opt.lr_decay_period = 50
    opt.lr_type = 'cosine_repeat_lr'
    opt.num_epochs=50
    opt.resume = 'model_29.pth'
    opt.debug = 0
    ### Other ###  
    opt.manual_seed = 704
    opt.log_batch_interval=10
    opt.log_checkpoint = 1
    opt.net_type = 'ResNet34DLAS_A'
    opt.pretrained = 'mcs2018'
    opt.classifier_type = 'linear'
    opt.loss_type= 'cce'
    opt.alpha_scheduler_type = None
    opt.nclasses = 2
    opt.fake_class_weight = 1
    opt.visdom_port = 8097
    
    opt.git_commit_sha = '3ab79d6c8ec9b280f5fbdd7a8a363a6191fd65ce' 
    opt.train_transform = tv.transforms.Compose([
            transforms.MergeItems(True, p=0.1),
            #transforms.LabelSmoothing(eps=0.1, p=0.2),
            transforms.CustomRandomRotation(30, resample=2),
            transforms.CustomResize((125,125)),
            tv.transforms.RandomApply([
                transforms.CustomCutout(1, 25, 75)],p=0.1),
            #transforms.CustomGaussianBlur(max_kernel_radius=3, p=0.2),
            transforms.CustomRandomResizedCrop(112, scale=(0.5, 1.0)),
            transforms.CustomRandomHorizontalFlip(),
            tv.transforms.RandomApply([
                transforms.CustomColorJitter(0.25,0.25,0.25,0.125)],p=0.2),
            transforms.CustomRandomGrayscale(p=0.1),
            transforms.CustomToTensor(),
            transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    opt.test_transform = tv.transforms.Compose([
            transforms.CustomResize((125,125)),
            transforms.CustomRotate(0),
            transforms.CustomRandomHorizontalFlip(p=0),
            transforms.CustomCrop((112,112), crop_index=0),
            transforms.CustomToTensor(),
            transforms.CustomNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    
    
    return opt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath', type=str, default = 'data/opts/', help = 'Path to save options')
    conf = parser.parse_args()
    opts = get_opts()
    save_dir = os.path.join(conf.savepath, opts.exp_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir,opts.exp_name + '_' + 'fold{0}'.format(opts.fold) + '_' + opts.task_name+'.opt')
    torch.save(opts, filename)
    print('Options file was saved to '+filename)
