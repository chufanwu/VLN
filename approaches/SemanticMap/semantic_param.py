import argparse
from time import strftime as tt


def parse_all_args():
    parser = argparse.ArgumentParser(description='Training params')

    # ------------------------------Others------------------------------#	parser.add_argument('--timestampLaunch',			type = str, 	default = tt("%Y%m%d") + '-' + tt("%H%M%S") ,		help = '')
    parser.add_argument('--logPath', type=str, default='', help='')
    parser.add_argument('--imageRootPath', type=str, default='', help='')
    parser.add_argument('--checkPoint', type=str, default='', help='')

    # ----------------------------Training------------------------------#
    parser.add_argument('--epoch', type=int, default=1, help='')
    parser.add_argument('--batchSize', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--batchPerlossDisplay', type=int, default=1, help='')
    parser.add_argument('--batchPerVisualize', type=int, default=1, help='')
    parser.add_argument('--batchPerPlot', type=int, default=1, help='')
    parser.add_argument('--saveEpoch', type=int, default=1, help='')

    args = parser.parse_args()

    return args
