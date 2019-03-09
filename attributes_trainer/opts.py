import argparse, os, json
import torch
import torchvision as tv
from utils import transforms

def get_opts():
    opt = argparse.Namespace()
    
    opt.task_name = 'umdfaces_cce_resnext50'
    opt.exp_name = 'umdfaces_exp3'
    opt.fold = 1
    opt.data_root = '/ssd/a.parkin/'
    opt.data_list = '/media3/a.parkin/media/FaceDatasets/lists/umdfaces_list/'
    
    opt.out_root = '/media3/a.parkin/media/FaceDatasets/opts/'
    opt.out_path = os.path.join(opt.out_root,opt.exp_name,'fold{fold_n}'.format(fold_n=opt.fold))
    
    ### Dataloader options ###
    opt.nthreads = 32
    opt.batch_size = 256 #280
    opt.ngpu = 4

    ### Learning ###
    opt.optimizer_name = 'SGD'
    opt.lr = 0.1
    opt.lr_decay_lvl = 0.1
    opt.lr_decay_period = 30
    opt.lr_type = 'step_lr'
    opt.num_epochs=120
    opt.resume = ''
    opt.debug = 0
    ### Other ###  
    opt.manual_seed = 42
    opt.log_batch_interval=10
    opt.log_checkpoint = 1
    opt.net_type = 'ResNext50'
    opt.pretrained = None
    opt.loss_type='arc_margin'
    opt.nclasses = 8277
    opt.fake_class_weight = 1
    opt.visdom_port = 8097
    
    opt.git_commit_sha = '3ab79d6c8ec9b280f5fbdd7a8a363a6191fd65ce' 
    opt.train_transform = tv.transforms.Compose([
            #transforms.MergeItems(True, p=0.2),
            #transforms.LabelSmoothing(eps=0.1, p=0.2),
            tv.transforms.RandomRotation(30, resample=2),
            #tv.transforms.Resize((125,125)),
            #transforms.CustomGaussianBlur(max_kernel_radius=3, p=0.2),
            tv.transforms.RandomResizedCrop(112, scale=(0.5, 1.0)),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.RandomApply([
                tv.transforms.ColorJitter(0.5,0.5,0.5,0.25)],p=0.2),
            tv.transforms.RandomGrayscale(p=0.2),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    opt.test_transform = tv.transforms.Compose([
            #tv.transforms.Resize((125,125)),
            tv.transforms.RandomHorizontalFlip(p=0),
            tv.transforms.CenterCrop((112,112)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    
    
    return opt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath', type=str, default = '/media3/a.parkin/media/FaceDatasets/opts/', help = 'Path to save options')
    conf = parser.parse_args()
    opts = get_opts()
    save_dir = os.path.join(conf.savepath, opts.exp_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir,opts.exp_name + '_' + 'fold{0}'.format(opts.fold) + '_' + opts.task_name+'.opt')
    torch.save(opts, filename)
    print('Options file was saved to '+filename)
