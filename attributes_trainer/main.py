import argparse,json,random,os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision as tv

from trainer import Model
from opts import get_opts
def main():
    
    # Load options
    parser = argparse.ArgumentParser(description='Attribute Learner')
    parser.add_argument('--config', type=str, help = 'Path to config .opt file. Leave blank if loading from opts.py')
    
    conf = parser.parse_args()
    opt = torch.load(conf.config) if conf.config else get_opts()
    
    print('===Options==')
    d=vars(opt)
    for k in d.keys():
        print(k,':',d[k])

     
    # Fix seed
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)
    cudnn.benchmark = True
    
    # Create working directories
    try:
        os.makedirs(opt.out_path)
        os.makedirs(os.path.join(opt.out_path,'checkpoints'))
        os.makedirs(os.path.join(opt.out_path,'log_files'))
        print( 'Directory {} was successfully created.'.format(opt.out_path))
                   
    except OSError:
        print( 'Directory {} already exists.'.format(opt.out_path))
        pass
    
    
    # Training
    M = Model(opt)
    M.train()
    '''
    TODO: M.test()
    '''
    
if __name__ == '__main__':
    main()


