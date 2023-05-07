import torch
import os

#%% Hardware configuration
BLOCK_BITS=6
PAGE_BITS=12
TOTAL_BITS=64
BLOCK_NUM_BITS=TOTAL_BITS-BLOCK_BITS
#%% Input and labeling configuration
SPLIT_BITS=6
LOOK_BACK=9
PRED_FORWARD=128
DELTA_BOUND=128
#%% Fixed bitmap dimensions of (height=LOOK_BACK+1, width=BLOCK_NUM_BITS / SPLIT_BITS)
BITMAP_SIZE=2*DELTA_BOUND
image_size=(LOOK_BACK+1, BLOCK_NUM_BITS//SPLIT_BITS+1)
patch_size=(1, image_size[1])
num_classes=2*DELTA_BOUND
#%% Filter configuration
Degree=16
FILTER_SIZE=16

#%% Configuaration class to get data from environment variables
class Config:
    def __init__(self):
        self.batch_size = int(os.environ.get('BATCH_SIZE', 64))
        self.epochs = int(os.environ.get('EPOCHS', 100))
        self.lr = float(os.environ.get('LR', 0.001))
        self.early_stop = int(os.environ.get('EARLY_STOP', 10))
        self.gamma = float(os.environ.get('GAMMA', 0.7))
        self.step_size = int(os.environ.get('STEP_SIZE', 1))
        self.gpu_id = int(os.environ.get('GPU_ID', 0))
        self.channels = int(os.environ.get('CHANNELS', 1))
        self.alpha = float(os.environ.get('ALPHA', 0.5))
        self.image_size = image_size
        
def get_config():
    cf = Config()
    device = torch.device(f"cuda:{cf.gpu_id}" if torch.cuda.is_available() else "cpu")
    return cf, device

#%% Logger class
import logging
from logging import handlers
class Logger(object):
    
    def __init__(self):
        pass
    
    def set_logger(self, log_path):
        #if os.path.exists(log_path) is True:
        #    os.remove(log_path)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
    
        if not self.logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            self.logger.addHandler(file_handler)
    
            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(stream_handler)
