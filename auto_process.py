import argparse

import torch
from loguru import logger
from netspresso.compressor import ModelCompressor, Task, Framework

from utils import get_network_np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')

    """
        Compression arguments
    """
    parser.add_argument(
        "--compression_method",
        type=str,
        choices=["PR_L2", "PR_GM", "PR_NN", "PR_ID", "FD_TK", "FD_CP", "FD_SVD"],
        default="PR_L2"
    )
    parser.add_argument(
        "--recommendation_method",
        type=str,
        choices=["slamp", "vbmf"],
        default="slamp"
    )
    parser.add_argument(
        "--compression_ratio",
        type=int,
        default=0.5
    )
    parser.add_argument(
        "-m",
        "--np_email",
        help="NetsPresso login e-mail",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--np_password",
        help="NetsPresso login password",
        type=str,
    )

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

    """ 
        Model compression - recommendation compression 
    """
    logger.info("Compression step start.")
    
    compressor = ModelCompressor(email=args.np_email, password=args.np_password)

    UPLOAD_MODEL_NAME = args.net
    TASK = Task.IMAGE_CLASSIFICATION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = fx_model_path
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [32, 32]}] # input size fixed!
    model = compressor.upload_model(
        model_name=UPLOAD_MODEL_NAME,
        task=TASK,
        framework=FRAMEWORK,
        file_path=UPLOAD_MODEL_PATH,
        input_shapes=INPUT_SHAPES,
    )

    COMPRESSION_METHOD = args.compression_method
    RECOMMENDATION_METHOD = args.recommendation_method
    RECOMMENDATION_RATIO = args.compression_ratio
    COMPRESSED_MODEL_NAME = f'{UPLOAD_MODEL_NAME}_{COMPRESSION_METHOD}_{RECOMMENDATION_RATIO}'
    OUTPUT_PATH = COMPRESSED_MODEL_NAME + '.pt'
    compressed_model = compressor.recommendation_compression(
        model_id=model.model_id,
        model_name=COMPRESSED_MODEL_NAME,
        compression_method=COMPRESSION_METHOD,
        recommendation_method=RECOMMENDATION_METHOD,
        recommendation_ratio=RECOMMENDATION_RATIO,
        output_path=OUTPUT_PATH,
    )

    logger.info("Compression step end.")
