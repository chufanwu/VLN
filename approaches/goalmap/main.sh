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


python -u ./goalmap_main.py\
                    --epoch 100000\
                    --batchSize 2\
                    --lrCNN 1e-9\
                    --lrRele 1e-3\
                    --batchPerlossDisplay 2\
                    --batchPerVisualize 2\
                    --batchPerPlot 2\
                    --batchPerModelSave 200\
                    --logPath "$LOGPATH"\
                    --imageRootPath "/home/qiyuand/matterport3D"\
                    --checkPoint "/home/qiyuand/GoalMap/log/rele-sentence-withbn/models/Epoch-001-Batch-6200.pth.tar"\
                    > "$LOGFILE" 2>&1 &

