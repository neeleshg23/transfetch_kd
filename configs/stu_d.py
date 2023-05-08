from models.d import DenseNetStudent
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
#%% ResNet Teacher Model w/ 24M trainable params
dim=256
depth=6
heads=8
mlp_dim=256
channels=1
context_gamma=0.2
#%% Model Definition
model = DenseNetStudent(num_classes, channels)