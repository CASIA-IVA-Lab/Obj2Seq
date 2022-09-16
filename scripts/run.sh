CONFIG_FILE=$1
OUTPUT_DIR=$2
NGPUS=8
PORT=34221

python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT --use_env main.py \
 --cfg $CONFIG_FILE --output_dir $OUTPUT_DIR --auto_resume ${@:3} | tee $OUTPUT_DIR/screen.log

