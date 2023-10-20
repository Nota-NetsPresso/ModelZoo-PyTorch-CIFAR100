import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from netspresso.compressor import ModelCompressor, Task, Framework

from conf import settings
from utils import get_network_np, get_training_dataloader, get_test_dataloader, WarmUpLR


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type', choices=['mobilenetv2', 'repvgg', 'vgg16', 'resnet56', 'inceptionv3'])
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
        default=0.3
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

    """
        Fine-tuning arguments
    """
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')

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

    """ 
        Retrain model 
    """
    logger.info("Fine-tuning step start.")
    # model load
    net = torch.load(OUTPUT_PATH)

    # data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr*0.1, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.T_MAX, eta_min=0, verbose=False)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        net = net.cuda()
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # trainer
    best_acc = 0.0
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        # train
        start = time.time()
        net.train()
        for batch_index, (images, labels) in enumerate(cifar100_training_loader):

            if args.gpu:
                labels = labels.cuda()
                images = images.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

            last_layer = list(net.children())[-1]
            for name, para in last_layer.named_parameters():
                if 'weight' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

            logger.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))

            #update training loss for each iteration
            writer.add_scalar('Train/loss', loss.item(), n_iter)

            if epoch <= args.warm:
                warmup_scheduler.step()

        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

        finish = time.time()

        logger.info('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

        # eval
        start = time.time()
        net.eval()

        test_loss = 0.0 # cost function error
        correct = 0.0

        for (images, labels) in cifar100_test_loader:

            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = net(images)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        finish = time.time()
        if args.gpu:
            logger.info('GPU INFO.....')
            logger.info(torch.cuda.memory_summary(), end='')
        logger.info('Evaluating Network.....')
        logger.info('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(cifar100_test_loader.dataset),
            correct.float() / len(cifar100_test_loader.dataset),
            finish - start
        ))

        # add informations to tensorboard
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

        acc = correct.float() / len(cifar100_test_loader.dataset)

        # save model
        if not epoch % settings.SAVE_EPOCH:
            weights_path = os.path.join(checkpoint_path, f'{args.net}_{epoch}.pt')
            logger.info('saving weights file to {}'.format(weights_path))
            torch.save(net, weights_path)
        
        if best_acc < acc:
            best_acc = acc
            torch.save(net, os.path.join(checkpoint_path, 'best_ckpt.pt'))

    writer.close()
    logger.info("Fine-tuning step end.")

    """
        Export model to onnx
    """
    logger.info("Export model to onnx format step start.")

    net = torch.load(os.path.join(checkpoint_path, 'best_ckpt.pt'), map_location='cpu')
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(net, dummy_input, COMPRESSED_MODEL_NAME + '.onnx', 
                      verbose=True, input_names=['input'], output_names=['output'], opset_version=12)
    logger.info(f'=> saving model to {COMPRESSED_MODEL_NAME}.onnx')

    logger.info("Export model to onnx format step end.")
