import os
import numpy
import torch
import torchvision

class Dataset(torch.utils.data.Dataset):

    def __init__(self, path=r"C:\Users\Hilmi\Documents\GitHub\tiny-imagenet\train"):
        self.data = []
        folders = os.listdir(path)
        folders.sort()
        for i, folder in enumerate(folders):
            sub_path = f"{path}/{folder}/images"
            files = os.listdir(sub_path)
            files.sort()
            for file in files:
                self.data.append((f"{sub_path}/{file}", i, folder))
        self.project = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ConvertImageDtype(torch.float32)])

    @property
    def label(self):
        buffer = {}
        for _, i, label in self.data:
            buffer[label] = i
        return buffer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.project(torchvision.io.read_image(self.data[index][0]))
        label = torch.as_tensor(self.data[index][1])
        if image.shape[0] == 1:
            image = torch.cat([image] * 3)
        return image, label

class Validation(torch.utils.data.Dataset):

    def __init__(self, path=r"C:\Users\Hilmi\Documents\GitHub\tiny-imagenet\val"):
        self.data = []
        classes = Dataset().label
        with open(f"{path}/val_annotations.txt") as label:
            for line in label:
                data = line.strip("\n").split("\t")
                if data[0].endswith(".JPEG"):
                    self.data.append((f"{path}/images/{data[0]}", classes[data[1]], data[1]))
        self.project = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ConvertImageDtype(torch.float32)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.project(torchvision.io.read_image(self.data[index][0]))
        label = torch.as_tensor(self.data[index][1])
        if image.shape[0] == 1:
            image = torch.cat([image] * 3)
        return image + torch.randn_like(image) * numpy.random.rand() / 128, label

if __name__ == "__main__":
    dataset = Validation()
    image, label = dataset[0]
    print(dataset.data[0])
    print(image.shape, image.min(), image.max(), label)