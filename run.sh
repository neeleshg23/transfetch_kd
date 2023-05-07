#!/bin/bash
LoadTrace_ROOT="/home/neel/Desktop/LoadTraces/spec06"
OUTPUT_ROOT="/home/neel/Desktop/transfetch_kd/res"

VERSION="vit"
MODEL="v"

export GPU_ID=0
export BATCH_SIZE=256
export EPOCHS=50
export LR=0.0002
export EARLY_STOP=10
export GAMMA=0.1
export STEP_SIZE=20
export CHANNELS=1
export ALPHA=0.5

TRAIN=20
VAL=10
TEST=10
SKIP=1

app_list=(473.astar-s0.txt.xz)

TRAIN_WARM=$TRAIN
TRAIN_TOTAL=$(($TRAIN + $VAL)) 

TEST_WARM=$TRAIN_WARM
TEST_TOTAL=$(($TRAIN+$TEST)) 

echo "TRAIN/VAL/TEST/SKIP: "$TRAIN"/"$VAL"/"$TEST"/"$SKIP

VERSION_TCH=$VERSION"_tch"
VERSION_STU=$VERSION"_stu"

mkdir $OUTPUT_ROOT
mkdir $OUTPUT_ROOT/$VERSION_TCH
mkdir $OUTPUT_ROOT/$VERSION_TCH/train

mkdir $OUTPUT_ROOT/$VERSION_STU
mkdir $OUTPUT_ROOT/$VERSION_STU/train

#for app1 in `ls $LoadTrace_ROOT`; do
for app1 in ${app_list[*]}; do
	echo $app1
	file_path=$LoadTrace_ROOT/${app1}
    tch_model_path=$OUTPUT_ROOT/$VERSION_TCH/train/${app1}.model.pth
    stu_model_path=$OUTPUT_ROOT/$VERSION_STU/train/${app1}.model.pth
	#app2=${app1%%.txt*}.trace.xz

    #python train_tch.py $MODEL $file_path $tch_model_path $TRAIN_WARM $TRAIN_TOTAL $SKIP
    python train_stu.py $MODEL $file_path $tch_model_path $stu_model_path $TRAIN_WARM $TRAIN_TOTAL $SKIP

	echo "done for app "$app1
done

