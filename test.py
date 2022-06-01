from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self) -> None:
        self.data=