set -ex
# models
MODEL='bicycle_gan'
RESULTS_DIR='./results/rrdb_test_bird_0'
#CHECKPOINTS_DIR='../checkpoints/n02279972_BGAN/n02279972_BGAN_bicycle_gan/'
#CHECKPOINTS_DIR='./checkpoints/n01531178_BGAN_x8/'

# dataset
CLASS='Bird_RRDB_BGAN_x8_0'
CHECKPOINTS_DIR='./checkpoints'
NZ=5
NO_FLIP='--no_flip'
DIRECTION='AtoB'
LOAD_SIZE=256
CROP_SIZE=256
INPUT_NC=3
NITER=200
NITER_DECAY=20
SAVE_EPOCH=5
NAME=${CLASS}_${MODEL}

OPT=./train_bird_x8.json
G_NET='IMRRDB_net'

# misc
GPU_ID=3  # gpu id
NUM_TEST=20 # number of input images duirng test
NUM_SAMPLES=50 # number of samples per input images
DISPLAY_ID=0

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test_rrdb.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./checkpoints/${CLASS} \
  --name ${NAME} \
  --nz ${NZ} \
  --direction ${DIRECTION} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --no_flip \
  --gpu_ids ${GPU_ID} \
  --opt ${OPT} \
  --netG ${G_NET} \
  --upsample bilinear
