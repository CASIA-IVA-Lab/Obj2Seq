NGPUS=1
PORT=34224

python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT --use_env main.py \
 --auto_resume --opts DATA.num_workers 0 $@
