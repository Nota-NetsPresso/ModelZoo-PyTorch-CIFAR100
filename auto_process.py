import argparse

import torch
from loguru import logger

from utils import get_network_np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()

    # Load pretrained model
    net = get_network_np(args).cpu()

    """ 
        Convert model to fx 
    """
    logger.info("Model to fx graph start.")

    _graph = torch.fx.Tracer().trace(net)
    traced_model = torch.fx.GraphModule(net, _graph)
    fx_model_path = args.net + '_fx.pt'
    torch.save(traced_model, './' + fx_model_path)
    logger.info(f"generated model to compress model {fx_model_path}")

    logger.info("Model to fx graph end.")
