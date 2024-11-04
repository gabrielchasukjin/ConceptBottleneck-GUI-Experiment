import torch
import json
from src.model import ConceptBottleneckCNN
from src.train import train
from src.test import test
from src.utils import get_data_loaders
from torch.optim import Adam

def main():
    with open('config/config.json', 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    train_loader, test_loader = get_data_loaders(config["batch_size"], config["test_batch_size"])

    model = ConceptBottleneckCNN(config["num_concepts"]).to(device)
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(1, config["epochs"] + 1):
        train(model, device, train_loader, optimizer, epoch, config["log_interval"])
        test(model, device, test_loader)

if __name__ == '__main__':
    main()
