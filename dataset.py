from PIL import Image
from torch.utils.data import Dataset
class CustomDataset(Dataset):
  def __init__(self,image_paths, labels, transform=None, phase='train'):
    self.image_paths = image_paths
    self.labels = labels
    self.transform = transform[phase]

  def __len__(self):
    return len(self.image_paths)
  
  def __getitem__(self, index):
    image = Image.open(self.image_paths[index])
    label = self.labels[index]

    if self.transform:
      image = self.transform(image)

    return image, label