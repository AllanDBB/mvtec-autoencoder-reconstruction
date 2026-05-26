from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MVTecDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path, label_int, defect_type_str, class_str)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, defect_type, cls = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, defect_type, cls


class MVTecDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, classes, image_size=128, batch_size=32,
                 num_workers=2, train_val_split=0.85):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def setup(self, stage=None):
        all_train_samples = []
        for cls in self.classes:
            good_dir = self.data_dir / cls / 'train' / 'good'
            for img_path in sorted(good_dir.glob('*.png')):
                all_train_samples.append((str(img_path), 0, 'good', cls))

        n_train = int(len(all_train_samples) * self.train_val_split)
        self.train_samples = all_train_samples[:n_train]
        self.val_samples = all_train_samples[n_train:]

        all_test_samples = []
        for cls in self.classes:
            test_dir = self.data_dir / cls / 'test'
            for defect_dir in sorted(test_dir.iterdir()):
                label = 0 if defect_dir.name == 'good' else 1
                for img_path in sorted(defect_dir.glob('*.png')):
                    all_test_samples.append((str(img_path), label, defect_dir.name, cls))

        self.train_dataset = MVTecDataset(self.train_samples, self.transform)
        self.val_dataset = MVTecDataset(self.val_samples, self.transform)
        self.test_dataset = MVTecDataset(all_test_samples, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                         shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=self.num_workers)
