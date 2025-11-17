# conda activate your_env_name

HERE=$(pwd)

now=`date '+%Y%m%d_%H%M%S'`
JOBNAME=sqa_Qwen_LoRA_${now}
LOGDIR=$HERE/logs/$JOBNAME

export WANDB_API_KEY=73f8ff40bb7f8589e9bd1f476196a896f662cdfa # sqa's wandb key

mkdir -p ${LOGDIR}
# sudo chmod 777 -R ${LOGDIR}
echo 'Log dir: '$LOGDIR

echo 'login wandb'
python -m wandb login $WANDB_API_KEY
sleep 1
python -m wandb login

python main.py --workdir=${LOGDIR} --config=configs/load_config.py:run