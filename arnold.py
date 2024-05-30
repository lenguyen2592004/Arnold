import argparse
import os
import vizdoom
from src.utils import get_dump_path
from src.logger import get_logger
from src.args import parse_game_args
import inspect

def my_function(arg1, *, arg2, arg3=None):  # Example function with keyword-only arguments and annotations
    pass

# Get function signature (including keyword-only args and annotations)
signature = inspect.signature(my_function)

# Access information about the parameters
for param in signature.parameters.values():
    print(param.name, param.kind, param.default)  # Output: arg1 POSITIONAL_OR_KEYWORD <class 'inspect._empty'>, arg2 KEYWORD_ONLY <class 'inspect._empty'>, arg3 KEYWORD_ONLY None

parser = argparse.ArgumentParser(description='Arnold runner')
parser.add_argument("--main_dump_path", type=str, default="./dumped",
                    help="Main dump path")
parser.add_argument("--exp_name", type=str, default="default",
                    help="Experiment name")
args, remaining = parser.parse_known_args()
assert len(args.exp_name.strip()) > 0

# create a directory for the experiment / create a logger
dump_path = get_dump_path(args.main_dump_path, args.exp_name)
logger = get_logger(filepath=os.path.join(dump_path, 'train.log'))
logger.info('========== Running DOOM ==========')
logger.info('Experiment will be saved in: %s' % dump_path)

# load DOOM
parse_game_args(remaining + ['--dump_path', dump_path])
