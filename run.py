import torch
torch.cuda.empty_cache()
import numpy as np
import os
from model.train import AnomalyTrainer
import argparse

os.environ['CUDE_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_global_config(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def distribute(rank, world_size, config):
    setup_global_config(rank, world_size)
    trainer = AnomalyTrainer(rank, config)
    trainer.train_distributed()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_python.yaml')
    parser.add_argument('--mode', type=str, default='ood')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test_main_task', dest='test_main_task', action='store_true')
    parser.add_argument('--test_baseline_metrics', dest='test_baseline_metrics', action='store_true')
    parser.add_argument('--ddp', dest='ddp', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_path', type=str, default=None)
    parser.set_defaults(train=False)
    parser.set_defaults(test=False)
    parser.set_defaults(ddp=False)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.train:
        trainer = AnomalyTrainer(args.gpu, args.config, args.seed)
        trainer.train()

    elif args.test_main_task:
        trainer = AnomalyTrainer(args.gpu, args.config, args.seed)
        trainer.test_main_task(args.model_path)

    else:
        trainer = AnomalyTrainer(args.gpu, args.config, args.seed)
        trainer.test_baseline_metrics(args.model_path)