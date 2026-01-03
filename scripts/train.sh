PORT=5678
GPU='0'
NUM_POINTS=121
IMG_SIZE=512
ENCODER='swin_tiny'
#TRAINSET='MambaVOS_swin_tiny_pretrain'
#EPOCH=500
TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=4
THRESHOLD=0.5
#TRAIN_ROOT='/home/qdk/code/MambaVOS-master/dataset/TrainSet/YoutubeVOS'
SAVE_ROOT='./exp_logs'
RESTORE_FROM='None'
INFER_SAVE=$SAVE_ROOT'/tmp_results'
#SAVE_PATH=$SAVE_ROOT'/'$TRAINSET'/'
NUM_GPUS=`echo $GPU | awk -F ',' '{print NF}'`
#CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port $PORT tools/train.py \
#  	--gpu $GPU \
#  	--num_points $NUM_POINTS \
#  	--img_size $IMG_SIZE \
#  	--threshold $THRESHOLD \
#  	--epoch $EPOCH \
#  	--train_batchsize $TRAIN_BATCH_SIZE \
#  	--val_batchsize $VAL_BATCH_SIZE \
#  	--train_root $TRAIN_ROOT \
#  	--trainset $TRAINSET \
#  	--infer_save $INFER_SAVE \
#  	--save_path $SAVE_PATH \
#  	--restore_from $RESTORE_FROM \
#  	--encoder $ENCODER

#RESTORE_FROM=$SAVE_PATH'best.pth'
EPOCH=300
TRAIN_ROOT='/home/qdk/code/MambaVOS-master/dataset/TrainSet/DAVIS_FBMS'
TRAINSET='MambaVOS_swin_tiny_finetune'
SAVE_PATH=$SAVE_ROOT'/'$TRAINSET'/'
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port $PORT tools/train.py \
 	--gpu $GPU \
 	--num_points $NUM_POINTS \
 	--img_size $IMG_SIZE \
 	--threshold $THRESHOLD \
 	--epoch $EPOCH \
 	--train_batchsize $TRAIN_BATCH_SIZE \
 	--val_batchsize $VAL_BATCH_SIZE \
 	--train_root $TRAIN_ROOT \
 	--trainset $TRAINSET \
 	--infer_save $INFER_SAVE \
 	--save_path $SAVE_PATH \
 	--restore_from $RESTORE_FROM \
 	--encoder $ENCODER
