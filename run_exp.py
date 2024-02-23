import os
import sys
import torch
import logging
import traceback
import numpy as np
from pprint import pprint

from runner import *
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config
torch.set_printoptions(profile='full')


def main():
  args = parse_arguments()
  config = get_config(args.config_file, is_test=(args.test or args.test_completion))
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  config.use_gpu = config.use_gpu and torch.cuda.is_available()

  ### TO-DO: Move to a separate function
  ### Setting some new flags here to work with older config files
  if not hasattr(config.dataset, "has_node_feat"):
    config.dataset.has_node_feat = False
  if not hasattr(config.dataset, "has_sub_nodes"):
    config.dataset.has_sub_nodes = False
  if not hasattr(config.dataset, "has_start_node"):
    config.dataset.has_start_node = False
  if not hasattr(config.dataset, "has_stop_node"):
    config.dataste.has_stop_node = False
    
  if config.dataset.has_sub_nodes:
    config.dataset.has_node_feat = True
    config.dataset.has_start_node = False ## Subnodes don't have start node logic implemented

  if config.dataset.has_start_node:
    config.model.max_num_nodes += 1
  if config.dataset.has_stop_node:
    config.model.max_num_nodes += 1


  # log info
  log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
  logger = setup_logging(args.log_level, log_file)
  logger.info("Writing log file to {}".format(log_file))
  logger.info("Exp instance id = {}".format(config.run_id))
  logger.info("Exp comment = {}".format(args.comment))
  logger.info("Config =")
  print(">" * 80)
  pprint(config)
  print("<" * 80)

  # Run the experiment
  # args.test = True #Comment while training
  # args.test_completion = True
  try:
    runner = eval(config.runner)(config)
    if not (args.test or args.test_completion):
      runner.train()
    elif args.test_completion:
      runner.test_completion()
    else:
      runner.test()
  except:
    logger.error(traceback.format_exc())

  sys.exit(0)


if __name__ == "__main__":
  main()