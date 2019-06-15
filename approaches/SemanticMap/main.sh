#!/bin/bash
OMP_NUM_THREADS=10
export OMP_NUM_THREADS
#

DIRNAME=`date '+%Y-%m-%d-%H-%M-%S'`
LOGPATH="./log/$DIRNAME"
LOGFILE="$LOGPATH/log.txt"
if [ ! -d "$LOGPATH" ];then
mkdir "$LOGPATH"
fi


python -u ./semantic_main.py\
                    --epoch 100\
                    --batchSize 100\
                    --lr 1e-4\
                    --batchPerlossDisplay 1\
                    --batchPerVisualize 10\
                    --batchPerPlot 10\
                    --saveEpoch 1\
                    --logPath "$LOGPATH"\
                    --imageRootPath "/home/qiyuand/matterport3D"\
                    --checkPoint ""\
                    > "$LOGFILE" 2>&1 &
