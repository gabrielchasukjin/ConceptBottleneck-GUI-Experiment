import torch
import torch.nn.functional as F
import torch.nn as nn

def generate_concept_labels(labels):
    has_loop, num_strokes, is_symmetric, has_diagonal, has_vertical, has_horizontal, is_circular = [], [], [], [], [], [], []
    for label in labels:
        digit = label.item()
        has_loop.append(1.0 if digit in [0, 6, 8, 9] else 0.0)
        num_strokes.append(1.0 if digit in [1] else 2.0 if digit in [2, 3, 5, 7] else 3.0)
        is_symmetric.append(1.0 if digit in [0, 3, 8] else 0.0)
        has_diagonal.append(1.0 if digit in [2, 4, 7] else 0.0)
        has_vertical.append(1.0 if digit in [1, 4] else 0.0)
        has_horizontal.append(1.0 if digit in [5, 7] else 0.0)
        is_circular.append(1.0 if digit in [0, 6, 8, 9] else 0.0)
    
    return torch.tensor([has_loop, num_strokes, is_symmetric, has_diagonal, has_vertical, has_horizontal, is_circular], dtype=torch.float).T

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        concept_labels = generate_concept_labels(target).to(device)

        optimizer.zero_grad()
        concepts_pred, logits = model(data)

        # Concept losses
        binary_concept_indices = [0, 2, 3, 4, 5, 6]
        concept_loss_binary = sum(bce_loss(concepts_pred[:, i], concept_labels[:, i]) for i in binary_concept_indices)
        concept_loss_strokes = mse_loss(concepts_pred[:, 1], concept_labels[:, 1])
        concept_loss = concept_loss_binary + concept_loss_strokes

        # Classification loss
        classification_loss = F.cross_entropy(logits, target)

        # Total loss
        total_loss = concept_loss + classification_loss
        total_loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_loss.item():.6f}')
