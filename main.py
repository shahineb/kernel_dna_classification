import os
import argparse
import logging
import importlib
from utils import DataLoader
from src.kernels import kernels
from src.classifiers import classifiers

# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logger = logging.getLogger("Kernelito")


def run(args):

    has_effect = False

    if args:
        try:
            dataloader = DataLoader.DataLoader()
            logger.info("Preparing kernels {} and classifier {} ...".format(args.kernel, args.classifier))
            kernel = kernels.choose(args.kernel)
            model = classifiers.choose(args.classifier)()
            task_name = "src.{}".format(args.task)
            logger.info("Running script at {}".format(task_name))
            task = importlib.import_module(task_name)
            task.run(logger, dataloader, kernel, model, args)

        except Exception as e:
            logger.exception(e)
            logger.error("Uhoh, the script halted with an error.")
    else:
        if not has_effect:
            logger.error(
                "Script halted without any effect. To run code, use command:\npython3 main.py <args>")

def path(d):
    try:
        d_ = os.path.join(os.getcwd(), 'src')
        d = os.path.join(d_, d)
        assert os.path.exists(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Run scripts for the MVA Kernel Methods Kaggle')
    argparser.add_argument('task', nargs="?", choices=['validation', 'inference'],
                        help='validation or inference')
    argparser.add_argument('--classifier', nargs="?", default='kernel-lr',  choices=['kernel-lr', 'kernel-svm'],
                        help='classifier')
    argparser.add_argument('--kernel', nargs="?", default='spectrum',  choices=['spectrum', 'substring'],
                        help='kernel for the classifier')
    argparser.add_argument('--val_size', nargs="?", type=float, default=0.1, help='validation size')
    argparser.add_argument('--rd', nargs="?", type=int, default=42, help='random seed')

    run(argparser.parse_args())