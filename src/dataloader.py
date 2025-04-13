import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import cv2
from pathlib import Path
import numpy as np


class CelebaDataset(Dataset):
    def __init__(self, path, img_size):
        super().__init__()

        self.img_paths = list(Path(path).glob("*.jpg"))
        self.img_size = img_size

    @staticmethod
    def transform(x):
        return (x / 127.5 - 1).astype(np.float32).transpose(2, 0, 1)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img

def get_dataloader(data_dir, img_size, batch_size, num_workers=0, ddp=False):
    dataset = CelebaDataset(data_dir, img_size)
    # print(f"Dataet::Num images={len(dataset)}")
    sampler = DistributedSampler(dataset) if ddp else None
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=(sampler is None), sampler=sampler)
    return dataloader

if __name__ == "__main__":
    dataset = CelebaDataset("/Users/petrushkovm/Downloads/celeba_hq_256", 256)
    print(len(dataset))

    img = dataset[10]
    print(img.shape, img.dtype, img.min(), img.max())

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # dataloader = DataLoader(dataset, batch_size=3, num_workers=0, shuffle=True)
    # print(f"Num batches = {len(dataloader)}")
    # for i, batch in enumerate(dataloader):
    #     if i < 5:
    #         print(batch.shape, batch.dtype)

    
    dataloader2 = get_dataloader("/Users/petrushkovm/Downloads/celeba_hq_256", img_size=256, batch_size=3, num_workers=0, ddp=False)
    print(f"Num batches = {len(dataloader2)}")
    for i, batch in enumerate(dataloader2):
        if i < 5:
            print(batch.shape, batch.dtype)