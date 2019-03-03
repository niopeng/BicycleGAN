set -ex
MODEL='bicycle_gan'
# dataset details
CLASS='n02279972_BGAN'  # facades, day2night, edges2shoes, edges2handbags, maps
NZ=8
NO_FLIP='--no_flip'
DIRECTION='BtoA'
LOAD_SIZE=256
CROP_SIZE=256
INPUT_NC=3
NITER=400
NITER_DECAY=400
SAVE_EPOCH=5

# training
GPU_ID=1
DISPLAY_ID=$((GPU_ID*1+1))
CHECKPOINTS_DIR=./checkpoints/${CLASS}/
NAME=${CLASS}_${MODEL}
NET_D=../checkpoints/${CLASS}/n02279972_BGAN_bicycle_gan/695_net_D.pth
NET_D2=../checkpoints/${CLASS}/n02279972_BGAN_bicycle_gan/695_net_D2.pth
NET_G=../checkpoints/${CLASS}/n02279972_BGAN_bicycle_gan/695_net_G.pth
NET_E=../checkpoints/${CLASS}/n02279972_BGAN_bicycle_gan/695_net_E.pth

# command
python ./train.py \
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
  --netD ${NET_D} \
  --netD2 ${NET_D2} \
  --netE ${NET_E} \
  --netG ${NET_G}
