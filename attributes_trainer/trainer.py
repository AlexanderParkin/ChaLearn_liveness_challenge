import torch
import torch.nn as nn
import numpy as np
import time, os
import math

from visdom import Visdom

import models, datasets, utils

class Model:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt.ngpu else "cpu")
        
        self.model, self.classifier = models.get_model(opt.net_type, 
                                                       opt.loss_type, 
                                                       opt.pretrained,
                                                       int(opt.nclasses))
        self.model = self.model.to(self.device)
        self.classifier = self.classifier.to(self.device)

        if opt.ngpu>1:
            self.model = nn.DataParallel(self.model)
            
        self.loss = models.init_loss(opt.loss_type)
        self.loss = self.loss.to(self.device)

        self.optimizer = utils.get_optimizer(self.model, self.opt)
        self.lr_scheduler = utils.get_lr_scheduler(self.opt, self.optimizer)

        self.train_loader = datasets.generate_loader(opt,'train') 
        self.test_loader = datasets.generate_loader(opt,'val')    
        
        self.epoch = 0
        self.best_epoch = False
        self.training = False
        self.state = {}
        

        self.train_loss = utils.AverageMeter()
        self.test_loss  = utils.AverageMeter()
        self.batch_time = utils.AverageMeter()
        if self.opt.loss_type in ['cce', 'bce', 'mse', 'arc_margin']:
            self.test_metrics = utils.AverageMeter()
        else:
            self.test_metrics = utils.ROCMeter()

        self.best_test_loss = utils.AverageMeter()                    
        self.best_test_loss.update(np.array([np.inf]))

        self.visdom_log_file = os.path.join(self.opt.out_path, 'log_files', 'visdom.log')
        self.vis = Visdom(port = opt.visdom_port,
                          log_to_filename=self.visdom_log_file,
                          env=opt.exp_name + '_' + str(opt.fold))

        self.vis_loss_opts = {'xlabel': 'epoch', 
                              'ylabel': 'loss', 
                              'title':'losses', 
                              'legend': ['train_loss', 'val_loss']}

        self.vis_epochloss_opts = {'xlabel': 'epoch', 
                              'ylabel': 'loss', 
                              'title':'epoch_losses', 
                              'legend': ['train_loss', 'val_loss']}

    def train(self):
        
        # Init Log file
        if self.opt.resume:
            self.log_msg('resuming...\n')
            # Continue training from checkpoint
            self.load_checkpoint()
        else:
             self.log_msg()


        for epoch in range(self.epoch, self.opt.num_epochs):
            self.epoch = epoch
            
            '''
            if epoch < 0:
                for param in self.model.module.body.parameters():
                    param.requires_grad=False
            elif epoch == 0:
                for param in self.model.module.body.parameters():
                    param.requires_grad=True
            '''

            self.lr_scheduler.step()
            self.train_epoch()
            self.test_epoch()
            self.log_epoch()
            self.vislog_epoch()
            self.create_state()
            self.save_state()  
    
    def train_epoch(self):
        """
        Trains model for 1 epoch
        """
        self.model.train()
        self.classifier.train()
        self.training = True
        torch.set_grad_enabled(self.training)
        self.train_loss.reset()
        self.batch_time.reset()
        time_stamp = time.time()
        self.batch_idx = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            
            self.batch_idx = batch_idx
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            
            output = self.model(data)

            if isinstance(self.classifier, nn.Linear):
                output = self.classifier(output)
            else:
                output = self.classifier(output, target)

            if self.opt.loss_type == 'bce' or self.opt.loss_type == 'mse':
                target = target.float()
                loss_tensor = self.loss(output.squeeze(), target)
            else:
                loss_tensor = self.loss(output, target)

            loss_tensor.backward()   

            self.optimizer.step()

            self.train_loss.update(loss_tensor.item())
            self.batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            self.log_batch(batch_idx)
            self.vislog_batch(batch_idx)
            if self.opt.debug and (batch_idx==10):
                print('Debugging done!')
                break;
            
    def test_epoch(self):
        """
        Calculates loss and metrics for test set
        """
        self.training = False
        torch.set_grad_enabled(self.training)
        self.model.eval()
        self.classifier.eval()
        
        self.batch_time.reset()
        self.test_loss.reset()
        self.test_metrics.reset()
        time_stamp = time.time()
        
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            
            output = self.classifier(output)
            if self.opt.loss_type == 'bce' or self.opt.loss_type == 'mse':
                target = target.float()
                loss_tensor = self.loss(output.squeeze(), target)
            else:
                loss_tensor = self.loss(output, target)
            self.test_loss.update(loss_tensor.item())

            if self.opt.loss_type == 'cce':
                output = torch.nn.functional.softmax(output, dim=1)
            elif self.opt.loss_type.startswith('arc_margin'):
                output = torch.nn.functional.softmax(output, dim=1)
            elif self.opt.loss_type == 'bce':
                output = torch.sigmoid(output)

            metrics = self.calculate_metrics(output, target)
            self.test_metrics.update(metrics)

            self.batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            self.log_batch(batch_idx)
            #self.vislog_batch(batch_idx)
            if self.opt.debug and (batch_idx==10):
                print('Debugging done!')
                break;

        self.best_epoch = self.test_loss.avg < self.best_test_loss.val
        if self.best_epoch:
            # self.best_test_loss.val is container for best loss, 
            # n is not used in the calculation
            self.best_test_loss.update(self.test_loss.avg, n=0)
     
    def calculate_metrics(self, output, target):   
        """
        Calculates test metrix for given batch and its input
        """
        batch_result = None
        
        t = target
        o = output
            
        if self.opt.loss_type == 'bce':
            binary_accuracy = (t.byte()==(o>0.5)).float().mean(0).cpu().numpy()  
            batch_result = binary_accuracy
        elif self.opt.loss_type =='mse':
            mean_average_error = torch.abs(t-o.squeeze()).mean(0).cpu().numpy()
            batch_result = mean_average_error
        elif self.opt.loss_type == 'cce' or self.opt.loss_type == 'arc_margin':
            top1_accuracy = (torch.argmax(o, 1)==t).float().mean().item()
            batch_result = top1_accuracy
        else:
            raise Exception('This loss function is not implemented yet')
                
        return batch_result  

    
    def log_batch(self, batch_idx):
        if batch_idx % self.opt.log_batch_interval == 0:
            cur_len = len(self.train_loader) if self.training else len(self.test_loader)
            cur_loss = self.train_loss if self.training else self.test_loss
            
            output_string = 'Train ' if self.training else 'Test '
            output_string +='Epoch {}[{:.2f}%]: [{:.2f}({:.3f}) s]\t'.format(self.epoch,
                                                                          100.* batch_idx/cur_len, self.batch_time.val,self.batch_time.avg)
            
            loss_i_string = 'Loss: {:.5f}({:.5f})\t'.format(cur_loss.val, cur_loss.avg)
            output_string += loss_i_string
            
            print(output_string)
    
    def vislog_batch(self, batch_idx):
        loader_len = len(self.train_loader) if self.training else len(self.test_loader)
        cur_loss = self.train_loss if self.training else self.test_loss
        loss_type = 'train_loss' if self.training else 'val_loss'
        
        x_value = self.epoch + batch_idx / loader_len
        y_value = cur_loss.val
        self.vis.line([y_value], [x_value], 
                        name=loss_type, 
                        win='losses', 
                        update='append')
        self.vis.update_window_opts(win='losses', opts=self.vis_loss_opts)
    
    def log_msg(self, msg=''):
        mode = 'a' if msg else 'w'
        f = open(os.path.join(self.opt.out_path, 'log_files', 'train_log.txt'), mode)
        f.write(msg)
        f.close()
             
    def log_epoch(self):
        """ Epoch results log string"""
        out_train = 'Train: '
        out_test = 'Test:  '
        loss_i_string = 'Loss: {:.5f}\t'.format(self.train_loss.avg)
        out_train += loss_i_string
        loss_i_string = 'Loss: {:.5f}\t'.format(self.test_loss.avg)
        out_test += loss_i_string
            
        out_test+='\nTest:  '
        out_test+= '{0}\t{1:.4f}\t'.format(self.opt.loss_type, self.test_metrics.avg)
            
        is_best = 'Best ' if self.best_epoch else ''
        out_res = is_best+'Epoch {} results:\n'.format(self.epoch)+out_train+'\n'+out_test+'\n'
        
        print(out_res)
        self.log_msg(out_res)
        

    def vislog_epoch(self):
        x_value = self.epoch
        self.vis.line([self.train_loss.avg], [x_value], 
                        name='train_loss', 
                        win='epoch_losses', 
                        update='append')
        self.vis.line([self.test_loss.avg], [x_value], 
                        name='val_loss', 
                        win='epoch_losses', 
                        update='append')
        self.vis.update_window_opts(win='epoch_losses', opts=self.vis_epochloss_opts)


    ''' LEGACY CODE '''
    '''
    def adjust_lr(self):
        if self.opt.lr_type == 'step_lr':
            Set the LR to the initial LR decayed by lr_decay_lvl every lr_decay_period epochs
            lr = self.opt.lr * (self.opt.lr_decay_lvl ** ((self.epoch+1) // self.opt.lr_decay_period))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.opt.lr_type == 'cosine_lr':
            Cosine LR by d.kireev@visionlabs.ru and d.nekhaev@visionlabs.ru
            n_batches = len(self.train_loader)
            t_total = self.opt.num_epochs * n_batches
            t_cur = ((self.epoch) % self.opt.num_epochs) * n_batches
            t_cur += self.batch_idx
            lr_scale = 0.5 * (1 + math.cos(math.pi * t_cur / t_total))
            lr_scale_prev = 0.5 * (1 + math.cos(
                math.pi * np.clip((t_cur - 1), 0, t_total) / t_total))
            lr_scale_change = lr_scale / lr_scale_prev
            self.lr *= lr_scale_change
            if self.batch_idx % self.opt.log_batch_interval == 0 and self.batch_idx == 0:
                print (f'LR: {self.lr:.4f}')
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        else:
            raise Exception('Unexpected lr type') 
    '''
                    
    def create_state(self):
        self.state = {       # Params to be saved in checkpoint
                'epoch' : self.epoch,
                'model_state_dict' : self.model.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'best_test_loss' : self.best_test_loss,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
            }
    
    def save_state(self):
        if self.opt.log_checkpoint == 0:
                self.save_checkpoint('checkpoint.pth')
        else:
            if (self.epoch % self.opt.log_checkpoint == 0):
                self.save_checkpoint('model_{}.pth'.format(self.epoch)) 
                  
    def save_checkpoint(self, filename):     # Save model to task_name/checkpoints/filename.pth
        fin_path = os.path.join(self.opt.out_path,'checkpoints', filename)
        torch.save(self.state, fin_path)
        if self.best_epoch:
            best_fin_path = os.path.join(self.opt.out_path, 'checkpoints', 'model_best.pth')
            torch.save(self.state, best_fin_path)
           

    def load_checkpoint(self):                            # Load current checkpoint if exists
        fin_path = os.path.join(self.opt.out_path,'checkpoints',self.opt.resume)
        if os.path.isfile(fin_path):
            print("=> loading checkpoint '{}'".format(fin_path))
            checkpoint = torch.load(fin_path, map_location=lambda storage, loc: storage)
            self.epoch = checkpoint['epoch'] + 1
            self.best_test_loss = checkpoint['best_test_loss']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            print("=> loaded checkpoint '{}' (epoch {})".format(self.opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.opt.resume))

        if os.path.isfile(self.visdom_log_file):
                self.vis.replay_log(log_filename=self.visdom_log_file)
            
