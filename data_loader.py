import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

from config import get_config
from preprocessing import read_load_trace_data, preprocessing, preprocessing_gen

cf, device = get_config()
batch_size = cf.batch_size

class MAPDataset(Dataset):
    def __init__(self, df):
        self.past=list(df["past"].values)
        self.future=list(df["future"].values)

    def __getitem__(self, idx):
        past = self.past[idx]
        future = self.future[idx]
        return [past,future]

    def __len__(self):
        return len(self.past)
    
    
    def collate_fn(self, batch):
        past_b = [x[0] for x in batch]
        future_b = [x[1] for x in batch]
#        data=rearrange(past_b, '(b h c) w-> b c h w',c=cf.channels,b=batch,h=cf.image_size[0],w=cf.image_size[1])
#        print(np.array(past_b).shape)
        data=rearrange(np.array(past_b), '(b c) h w-> b c h w',c=cf.channels,h=cf.image_size[0],w=cf.image_size[1])

        past_tensor=torch.Tensor(data).to(device)
        
        future_tensor=torch.Tensor(future_b).to(device)

        
        return past_tensor, future_tensor

def data_generator(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM,only_val=False):
    if only_val==True:
        print("only validation")
        _, eval_data = read_load_trace_data(file_path, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
        df_test = preprocessing(eval_data)
        test_dataset = MAPDataset(df_test)
        
        #logging.info("-------- Dataset Build! --------")
        dev_dataloader=DataLoader(test_dataset,batch_size=cf.batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)
        
        return dev_dataloader, df_test
    else:
        print("train and validation")
        train_data, eval_data = read_load_trace_data(file_path, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
        df_train = preprocessing(train_data)
        df_test = preprocessing(eval_data)
        
        train_dataset = MAPDataset(df_train)
        test_dataset = MAPDataset(df_test)
        
        #logging.info("-------- Dataset Build! --------")
        train_dataloader=DataLoader(train_dataset,batch_size=cf.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
        dev_dataloader=DataLoader(test_dataset,batch_size=cf.batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)
        
        return train_dataloader, dev_dataloader, df_test
#%%

class MAPDataset_gen(Dataset):
    def __init__(self, df):
        self.past=list(df["past"].values)
    def __getitem__(self, idx):
        past = self.past[idx]

        return [past]

    def __len__(self):
        return len(self.past)
    
    def collate_fn(self, batch):
        
        past_b = [x[0] for x in batch]
        data=rearrange(np.array(past_b), '(b c) h w-> b c h w',c=cf.channels,h=cf.image_size[0],w=cf.image_size[1])
        past_tensor=torch.Tensor(data).to(device)
    
        return past_tensor


def data_generator_gen(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM):
    #file_path='/home/pengmiao/Disk/work/MLPrefComp/ChampSim/ML-DPC/LoadTraces/spec17/654.roms-s0.txt.xz'
    #model_save_path = "./results/spec17/654.roms-s0.model.pth"
    #TRAIN_NUM=10
    #log_path = model_save_path+".log"
    _, eval_data = read_load_trace_data(file_path, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    df_test = preprocessing_gen(eval_data)
    
    test_dataset = MAPDataset_gen(df_test)
    
    dev_dataloader=DataLoader(test_dataset,batch_size=cf.batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)
    
    return dev_dataloader,df_test