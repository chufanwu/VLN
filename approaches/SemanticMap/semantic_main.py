# from semantic_train import SemanticMapTask
from semantic_train_v2 import SemanticMapTask
from semantic_param import parse_all_args


if __name__ == '__main__':
    args = parse_all_args()
    Task = SemanticMapTask(args)
    Task.train()
