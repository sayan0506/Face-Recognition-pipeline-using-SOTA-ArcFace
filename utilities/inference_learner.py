import torch

class inference_face_learner(object):
  def __init__(self, config, model, data_loader, data_class_num):
    #print(f'Configureation-\n{config}')
    # if we want to load MobileFcaenet model
    if config.use_mobilefacenet:
      # model already loaded to device
      self.model = model
      self.load_state(config)
      print('MobilefaceNet model loaded!')
    else:
      print('Load different model!')

  def load_state(self, config):
    self.model.load_state_dict(torch.load(config.best_model))
