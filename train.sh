# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
# t2t-trainer --registry_help

PROBLEM=algorithmic_math_deepmind_all
MODEL=transformer
HPARAMS=transformer_base_v2

DATA_DIR=data1
TMP_DIR=tmp
VERSION=112M_in1_L256_share_embed_lr1e-4
TRAIN_DIR=train/$PROBLEM/$MODEL-$HPARAMS-$VERSION

# mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# # Generate data
# glibc_2.23_run.sh /mnt/lustre/mayukun/anaconda3/bin/python /mnt/lustre/mayukun/anaconda3/bin/t2t-datagen \
#   --data_dir=$DATA_DIR \
#   --tmp_dir=$TMP_DIR \
#   --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
srun --job-name=t2t_maths -p SenseVideo5 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=40 \
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --worker_gpu=8 \
  --train_steps=500000 \
  --hparams='num_hidden_layers=6,learning_rate=1e-4,learning_rate_schedule=constant*linear_warmup,learning_rate_constant=1e-4,clip_grad_norm=0.1,optimizer_adam_beta2=0.995,batch_size=8192,shared_embedding_and_softmax_weights=1'
