import torch
import tqdm
from train import Classifier
from dataset import Validation, Dataset

if __name__ == "__main__":
    model = Classifier()
    model.load_state_dict(torch.load("model.pth"))
    model = model.model.eval().cuda()

    dataset = Dataset()

    result = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset))):
            image, label = dataset[0]
            output = torch.argmax(model(image[None].cuda())[0]).cpu().item()
            result.append(1.0 if output == label else 0.0)
    result = torch.as_tensor(result)

    print(result.mean())