import torch
from torch import optim
import model as md
import os


class face_learner(object):
  def __init__(self, config, model, train_loader, train_class_num, valid_loader, valid_class_num):
    print(f'Configureation-\n{config}')
    # if we want to load MobileFcaenet model
    if config.use_mobilefacenet:
      # model already loaded to device
      self.model = model
      print('MobilefaceNet model generated!')
    else:
      print('Load the different model!')
    
    # defines the milestones where, the lr_scheduler will act
    self.milestones = config.milestones
    self.val_loader, self.val_class_num = valid_loader, valid_class_num
    self.loader, self.class_num = train_loader, train_class_num

    # helps to log summary of training using TensorboardX
    #self.writer = SummaryWriter(config.log_path)
    self.step = 0
    self.head = md.Arcface(embedding_size=config.embedding_size, classnum=self.class_num).to(config.device)

    print('Two model heads generated!')

    paras_only_bn, paras_wo_bn = self.seperate_bn_paras(self.model)   
    #print(len(paras_only_bn))
    #print(len(paras_wo_bn))
    # define optimizer for mobilefacenet
    if config.use_mobilefacenet:
      self.optimizer = optim.SGD([
                                  {'params': paras_wo_bn[:-1], 'weight_decay': 4e-05},# weight decay for the params
                                  {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-04},
                                  {'params': paras_only_bn}
      ], lr = config.lr, momentum = config.momentum)
    
    else:
      self.optimizer = optim.SGD([
                                  {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-04},
                                  {'params': paras_only_bn}
      ], lr = config.lr, momentum = config.momentum)
  
    print(self.optimizer)
    
    print('Optimizers generated')

  def seperate_bn_paras(self, modules):
    # if modules is not object or instantiated
    if not isinstance(modules, list):
      modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
      if 'model' in str(layer.__class__):
        continue
      if 'container' in str(layer.__class__):
        continue
      else:
        if 'batchnorm' in str(layer.__class__):
          # takes bin of parameters
          paras_only_bn.extend([*layer.parameters()])
        else:
          paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

  # load the state from saved model
  def load_state(self, config, fixed_str, from_save_folder = False, model_only = False):
    if from_save_folder:
      load_save_path = config.pretrained_save_path
    else:
      load_save_path = config.model_path
    self.model.load_state_dict(torch.load(os.path.join(load_save_path,'model_{}'.format(fixed_str))))
    if not model_only:
      self.head.load_state_dict(torch.load(os.path.join(load_save_path,'head_{}'.format(fixed_str))))
      self.optimizer.load_state_dict(torch.load(os.path.join(load_save_path,'optimizer_{}'.format(fixed_str))))
      
  # lr scheduler
  def schedule_lr(self):
    for params in self.optimizer.param_groups:
      params['lr']/=10
    print(self.optimizer)

  # save state of the model
  def save_state(self, config, loss, model_only = False):
    # save to model to config path
    save_path = config.save_path

    torch.save(
        self.model.state_dict(), os.path.join(save_path, 
        f'model_step_{self.step}_loss_{loss}.path'))
    
    if not model_only:
      torch.save(
          self.head.state_dict(), os.path.join(save_path, f'head_step_{self.step}_loss_{loss}.path')
      )

      torch.save(
          self.optimizer.state_dict(), os.path.join(save_path,
                                                    f'optimizer_step_{self.step}_loss_{loss}.path')
      )
  
    self.save_validated_model = tuple([os.path.join(save_path, f'model_step_{self.step}_loss_{loss}.path'),
                             os.path.join(save_path, f'head_step_{self.step}_loss_{loss}.path'),
                             os.path.join(save_path, f'optimizer_step_{self.step}_loss_{loss}.path')])
