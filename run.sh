# conda activate your_env_name

HERE=$(pwd)

now=`date '+%Y%m%d_%H%M%S'`
JOBNAME=sqa_Qwen_LoRA_${now}
LOGDIR=$HERE/logs/$JOBNAME

sudo mkdir -p ${LOGDIR}
sudo chmod 777 -R ${LOGDIR}
echo 'Log dir: '$LOGDIR

python main.py --workdir=${LOGDIR} --config=configs/load_config.py:run