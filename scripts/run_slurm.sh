TAG=`date +%m%d%H%M%S`
JOB_NAME="obj2seq"

PARTITION="name_of_partition"
NNODES=$1
NGPUS=`expr $NNODES \* 8`
CONFIG_FILE=$2
OUTPUT_DIR=$3

srun -p $PARTITION -n $NGPUS --gres=gpu:8 --ntasks-per-node=8 --job-name=${JOB_NAME} --cpus-per-task=5\
 python -u main.py --cfg $CONFIG_FILE --output_dir $OUTPUT_DIR --auto_resume ${@:4} 1>$OUTPUT_DIR/screen_${TAG}.log 2>&1 &