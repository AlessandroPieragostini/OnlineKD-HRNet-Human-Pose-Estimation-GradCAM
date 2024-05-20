import torch
import torchvision.transforms as transforms

import dataset

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    

def get_loader(cfg):
        train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
                transforms.Compose([
                transforms.ToTensor(),
                normalize]))
        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=cfg.TRAIN.SHUFFLE,
                num_workers=cfg.WORKERS,
                pin_memory=cfg.PIN_MEMORY)
        
        valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
                transforms.Compose([
                transforms.ToTensor(),
                normalize]))
        valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=False,
                num_workers=cfg.WORKERS,
                pin_memory=cfg.PIN_MEMORY)
        
        return train_dataset, train_loader, valid_dataset, valid_loader