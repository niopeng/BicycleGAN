set -ex
MODEL='bicycle_gan'
# dataset details
CLASS='Cicada_RRDB_BGAN_x8_4'  # facades, day2night, edges2shoes, edges2handbags, maps
NZ=5
NO_FLIP='--no_flip'
DIRECTION='AtoB'
LOAD_SIZE=256
CROP_SIZE=256
INPUT_NC=3
NITER=400
NITER_DECAY=400
SAVE_EPOCH=5

# training
GPU_ID=3
DISPLAY_ID=0
CHECKPOINTS_DIR=./checkpoints/${CLASS}/
NAME=${CLASS}_${MODEL}
NET_D=./pretrained_models/${CLASS}/latest_net_D.pth
NET_D2=./pretrained_models/${CLASS}/latest_net_D2.pth
NET_G=./pretrained_models/${CLASS}/latest_net_G.pth
NET_E=./pretrained_models/${CLASS}/latest_net_E.pth

OPT=./train_srim_cicada_x8.json
G_NET='IMRRDB_net'

# command
python ./train_rrdb.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --nz ${NZ} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --save_epoch_freq ${SAVE_EPOCH} \
  --use_dropout \
  --gpu_ids ${GPU_ID} \
  --lr 0.0001 \
  --opt ${OPT} \
  --netG ${G_NET} \
  --upsample bilinear
