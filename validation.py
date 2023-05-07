import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm 
import sys

import config
from configs.stu_d import model as stu_d
from configs.stu_r import model as stu_r
from configs.stu_v import model as stu_v
from configs.tch_d import model as tch_d
from configs.tch_r import model as tch_r
from configs.tch_v import model as tch_v
from threshold_throttling import threshold_throttleing

model = None

cf, device = config.get_config()

batch_size=cf.batch_size
epochs = cf.epochs
lr = cf.lr
gamma = cf.gamma

BLOCK_BITS=config.BLOCK_BITS
TOTAL_BITS=config.TOTAL_BITS
LOOK_BACK=config.LOOK_BACK
PRED_FORWARD=config.PRED_FORWARD

BLOCK_NUM_BITS=config.BLOCK_NUM_BITS
PAGE_BITS=config.PAGE_BITS
BITMAP_SIZE=config.BITMAP_SIZE
DELTA_BOUND=config.DELTA_BOUND
SPLIT_BITS=config.SPLIT_BITS
FILTER_SIZE=config.FILTER_SIZE

def set_model(model_name):
    global model
    if model_name == "stu_d":
        model = stu_d
    elif model_name == "stu_r":
        model = stu_r
    elif model_name == "stu_v":
        model = stu_v
    elif model_name == "tch_d":
        model = tch_d
    elif model_name == "tch_r":
        model = tch_r
    elif model_name == "tch_v":
        model = tch_v

def model_prediction(test_loader, test_df, model_save_path):#"top_k";"degree";"optimal"
    print("predicting")
    prediction=[]
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
    y_score=np.array([])
    for data,_ in tqdm(test_loader):
        output= model(data)
        #prediction.extend(output.cpu())
        prediction.extend(output.cpu().detach().numpy())
    test_df["y_score"]= prediction

    return test_df[['id', 'cycle', 'addr', 'ip','block_address','future', 'y_score']]

def evaluate(y_test,y_pred_bin):
    f1_score_res=f1_score(y_test, y_pred_bin, average='micro')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='micro')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='micro',zero_division=0)
    print("p,r,f1:",precision_score_res,recall_score_res,f1_score_res)
    return precision_score_res,recall_score_res,f1_score_res

##########################################################################################################
#%% New post_processing_delta_bitmap

def convert_hex(pred_block_addr):
    res=int(pred_block_addr)<<BLOCK_BITS
    res2=res.to_bytes(((res.bit_length() + 7) // 8),"big").hex().lstrip('0')
    return res2

def add_delta(block_address,pred_index):
    if pred_index<DELTA_BOUND:
        pred_delta=pred_index+1
    else:
        pred_delta=pred_index-BITMAP_SIZE
        
    return block_address+pred_delta


def post_processing_delta_filter(df):
    print("filtering")
    pred_array=np.stack(df["predicted"])
    pred_n_degree=pred_array
    
    df["pred_index"]=pred_n_degree.tolist()
    df=df.explode('pred_index')
    df=df.dropna()[['id', 'cycle', 'block_address', 'pred_index']]
    
    #add delta to block address
    df['pred_block_addr'] = df.apply(lambda x: add_delta(x['block_address'], x['pred_index']), axis=1)
    
    #filter
    que = []
    pref_flag=[]
    dg_counter=0
    df["id_diff"]=df["id"].diff()
    
    for index, row in df.iterrows():
        if row["id_diff"]!=0:
            que.append(row["block_address"])
            dg_counter=0
        pred=row["pred_block_addr"]
        if dg_counter<cf.Degree:
            if pred in que:
                pref_flag.append(0)
            else:
                que.append(pred)
                pref_flag.append(1)
                dg_counter+=1
        else:
            pref_flag.append(0)
        que=que[-FILTER_SIZE:]
    
    df["pref_flag"]=pref_flag
    df=df[df["pref_flag"]==1]
    df['pred_hex'] = df.apply(lambda x: convert_hex(x['pred_block_addr']), axis=1)
    df_res=df[["id","pred_hex"]]
    return df_res
    

def run_val(test_loader,test_df,file_path,model_save_path):
    print("Validation start")
    test_df=model_prediction(test_loader, test_df,model_save_path)

    df_thresh={}
    app_name=file_path.split("/")[-1].split("-")[0]
    val_res_path=model_save_path+".val_res.csv"
    
    df_res, threshold = threshold_throttleing(test_df,throttle_type="f1",optimal_type="micro")
    p,r,f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    df_thresh["app"],df_thresh["opt_th"],df_thresh["p"],df_thresh["r"],df_thresh["f1"]=[app_name],[threshold],[p],[r],[f1]
    
    df_res, _ = threshold_throttleing(test_df,throttle_type="fixed_threshold",threshold=0.5)
    p,r,f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    df_thresh["p_5"],df_thresh["r_5"],df_thresh["f1_5"]=[p],[r],[f1]
    
    pd.DataFrame(df_thresh).to_csv(val_res_path,header=1, index=False, sep=" ") #pd_read=pd.read_csv(val_res_path,header=0,sep=" ")
    print("Done: results saved at:", val_res_path)