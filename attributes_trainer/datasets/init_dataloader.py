from .init_dataset import ImageListDataset
import torch.utils.data
import os
import pandas as pd

def generate_loader(opt, split, inference_list = None):
    
    if split == 'train':
        current_transform = opt.train_transform
        current_shuffle = True
        sampler = None
        drop_last = True
        
    else:
        current_transform = opt.test_transform
        current_shuffle = False
        sampler = None
        drop_last = False
    
    data_list = os.path.join(opt.data_list, split + '_list.txt')
    data_root = opt.data_root    
    dataset = ImageListDataset(data_root = data_root,  data_list = data_list, transform=current_transform)

    assert dataset
    if split == 'train' and opt.fake_class_weight != 1:
        weights = [opt.fake_class_weight if x != 1 else 1.0 for x in dataset.df.label.values]
        num_samples = len(dataset)
        replacement = True
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement)
        current_shuffle = False
    if split == 'train' and len(dataset) % (opt.batch_size // opt.ngpu) < 32:
        drop_last = True

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = current_shuffle,
                                                 num_workers = int(opt.nthreads),sampler = sampler, pin_memory=True,
                                                 drop_last = drop_last)
    return dataset_loader
