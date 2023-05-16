import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary
import warnings
warnings.filterwarnings('ignore')

import config
from configs.tch_d import model as tch_d
from configs.tch_r import model as tch_r
from configs.tch_v import model as tch_v
from configs.stu_d import model as stu_d
from configs.stu_r import model as stu_r
from configs.stu_v import model as stu_v
from data_loader import data_generator
from validation import set_model, run_val

cf, device = config.get_config()

torch.manual_seed(100)

batch_size = cf.batch_size
epochs = cf.epochs
lr = cf.lr
gamma = cf.gamma
step_size = cf.step_size
early_stop = cf.early_stop
alpha = cf.alpha

model = None
teacher_model = None
optimizer = None
scheduler = None

soft_loss = nn.KLDivLoss(reduction="mean")
log = config.Logger()

def train(ep, train_loader, model_save_path):
    global steps
    epoch_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        student_preds = model(data)
        with torch.no_grad():
            teacher_preds = teacher_model(data)
        
        student_loss = F.binary_cross_entropy(student_preds, target, reduction='mean')
        distillation_loss = soft_loss(student_preds, teacher_preds)
        loss = alpha * student_loss + (1 - alpha) * distillation_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss/=len(train_loader)
    return epoch_loss


def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
            thresh=0.5
            #thresh=output.data.topk(pred_num)[0].min(1)[0].unsqueeze(1)
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /=  len(test_loader)
        return test_loss   

def run_epoch(epochs, loading, tch_model_save_path, model_save_path, train_loader, test_loader, lr):
    teacher_model.load_state_dict(torch.load(tch_model_save_path))
    log.logger.info("-------------Teacher Model Loaded------------")

    if loading==True:
        model.load_state_dict(torch.load(model_save_path))
        log.logger.info("-------------Student Model Loaded------------")
        
    best_loss=0
    early_stop=cf.early_stop
    for epoch in range(epochs):
        train_loss=train(epoch,train_loader,model_save_path)
        test_loss=test(test_loader)
        log.logger.info((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        if epoch == 0:
            best_loss=test_loss
        if test_loss<=best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss=test_loss
            log.logger.info("-------- Save Best Model! --------")
            early_stop=cf.early_stop
        else:
            early_stop-=1
            log.logger.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            log.logger.info("-------- Early Stop! --------")
            break

if __name__ == "__main__":
    option = sys.argv[1]

    if option == "d":
        model = stu_d
        teacher_model = tch_d
    elif option == "r":
        model = stu_r
        teacher_model = tch_r
    elif option == "v":
        model = stu_v
        teacher_model = tch_v
    
    model = model.to(device)
    teacher_model = teacher_model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    file_path = sys.argv[2]
    tch_model_save_path = sys.argv[3]
    model_save_path = sys.argv[4]
    TRAIN_NUM = int(sys.argv[5])
    TOTAL_NUM = int(sys.argv[6])
    SKIP_NUM = int(sys.argv[7])

    loading=False
    log_path=model_save_path+".log"
    log.set_logger(log_path)
    log.logger.info("%s"%file_path)
    
    log.logger.info(summary(model))
    
    train_loader, test_loader, test_df = data_generator(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    log.logger.info("-------------Data Proccessed------------")
    run_epoch(epochs, loading, tch_model_save_path, model_save_path, train_loader,test_loader,lr=cf.lr)
    set_model(f"stu_{option}")
    run_val(test_loader,test_df,file_path,model_save_path)