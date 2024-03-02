from .dataset import MVTecDataset,MVTec3DDataset,MVTec_3D_pretrained
from torch.utils.data import DataLoader


def build_dataset(datadir:str, texturedir:str, target:str, train:bool=True, to_memory:bool=False):
    dataset = MVTecDataset(
        datadir                = datadir,
        target                 = target, 
        train                  = train,
        to_memory              = to_memory,
        resize                 = (256, 256),
        texture_source_dir     = texturedir,
        structure_grid_size    = 8,
        transparency_range     = [0.15, 1.0],
        perlin_scale           = 6, 
        min_perlin_scale       = 0, 
        perlin_noise_threshold = 0.5
    )
    return dataset

def build_dataLoader(dataset, train: bool, batch_size: int = 6, num_workers: int = 1):
    dataloader = DataLoader(
        dataset,
        shuffle     = train,
        batch_size  = batch_size,
        num_workers = num_workers
    )

    return dataloader


def build_dataset3D(datadir:str, texturedir:str, target:str, train:bool=True, to_memory:bool=False):
    dataset = MVTec3DDataset(
        datadir                = datadir,
        target                 = target, 
        train                  = train,
        to_memory              = to_memory,
        resize                 = (256, 256),
        texture_source_dir     = texturedir,
        structure_grid_size    = 8,
        transparency_range     = [0.15, 1.0],
        perlin_scale           = 6, 
        min_perlin_scale       = 0, 
        perlin_noise_threshold = 0.5
    )
    return dataset

def build_dataLoader3D(dataset, train: bool, batch_size: int = 6, num_workers: int = 1):
    dataloader = DataLoader(
        dataset,
        shuffle     = train,
        batch_size  = batch_size,
        num_workers = num_workers
    )

    return dataloader



def build_dataset_3D_pretrained(datadir:str, texturedir:str, target:str):
    dataset = MVTec_3D_pretrained(
        datadir                = datadir,
        target                 = target, 
        resize                 = (256, 256),
        texture_source_dir     = texturedir,
        structure_grid_size    = 8,
        transparency_range     = [0.15, 1.0],
        perlin_scale           = 6, 
        min_perlin_scale       = 0, 
        perlin_noise_threshold = 0.5
    )
    return dataset


def build_dataLoader_3D_pretrained(dataset, batch_size = 4):
    dataLoader = DataLoader(
        dataset,
        shuffle = True,
        batch_size = batch_size
    )
    return dataLoader