# from semantic_train import SemanticMapTask
from goalmap_train_v4 import Task
from goalmap_param import parse_all_args


if __name__ == '__main__':
    args = parse_all_args()
    task = Task(args)
    task.train()
