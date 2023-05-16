#!/bin/bash
LoadTrace_ROOT="/home/neel/Desktop/LoadTraces/spec06"
OUTPUT_ROOT="/home/neel/Desktop/transfetch_kd/res"

VERSION="vit"
MODEL="v"

NUM_TCH=2

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

mkdir -p $OUTPUT_ROOT/$VERSION_TCH/train
mkdir -p $OUTPUT_ROOT/$VERSION_STU/train

for app1 in ${app_list[*]}; do
    echo "Processing: " $app1
    file_path=$LoadTrace_ROOT/${app1}

    echo "Decompressing file: " $file_path
    xz -d -k $file_path

    base_filename=${app1%.txt.xz}
    decompressed_file_path=$LoadTrace_ROOT/$base_filename.txt

    echo "Splitting the decompressed file into NUM_TCH parts"
    lines=$(wc -l <$decompressed_file_path)
    lines_per_part=$((lines / NUM_TCH))
    split -l $lines_per_part --numeric-suffixes $decompressed_file_path ${LoadTrace_ROOT}/${base_filename}_part_ --additional-suffix=.txt

    for part in $(seq 1 $NUM_TCH); do
        split_file_path="${LoadTrace_ROOT}/${base_filename}_part_$(printf "%02d" $((part - 1))).txt"
        split_file_compressed="${split_file_path}.xz"
        echo "Compressing part file: " $split_file_path
        xz -k -f $split_file_path

        tch_model_path=$OUTPUT_ROOT/$VERSION_TCH/train/${base_filename}_$(printf "%02d" $part)_of_$(printf "%02d" $NUM_TCH).model.pth
        echo "Training teacher model with file: " $split_file_compressed
        python train_tch.py $MODEL $split_file_compressed $tch_model_path $TRAIN_WARM $TRAIN_TOTAL $SKIP

        echo "Removing compressed part file: " $split_file_compressed
        rm $split_file_compressed
    done

    echo "Removing decompressed file and parts"
    rm $decompressed_file_path
    rm ${LoadTrace_ROOT}/${base_filename}_part_*

    # stu_model_path=$OUTPUT_ROOT/$VERSION_STU/train/${app1}.model.pth
    # python train_stu.py $MODEL $NUM_TCH $file_path $tch_model_path $stu_model_path $TRAIN_WARM $TRAIN_TOTAL $SKIP

    echo "done for app "$app1
done
