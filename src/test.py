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

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    concept_correct = 0
    total_concepts = 0

    bce_loss = nn.BCELoss(reduction='sum')
    mse_loss = nn.MSELoss(reduction='sum')

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            concept_labels = generate_concept_labels(target).to(device)
            concepts_pred, logits = model(data)

            # Losses
            binary_concept_indices = [0, 2, 3, 4, 5, 6]
            concept_loss_binary = sum(bce_loss(concepts_pred[:, i], concept_labels[:, i]) for i in binary_concept_indices)
            concept_loss_strokes = mse_loss(concepts_pred[:, 1], concept_labels[:, 1])
            concept_loss = concept_loss_binary + concept_loss_strokes
            classification_loss = F.cross_entropy(logits, target, reduction='sum')
            total_loss = concept_loss + classification_loss

            test_loss += total_loss.item()

            # Predictions
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Concept accuracy
            concept_pred_labels = concepts_pred.clone()
            for i in binary_concept_indices:
                concept_pred_labels[:, i] = (concepts_pred[:, i] > 0.5).float()
            concept_pred_labels[:, 1] = torch.round(concepts_pred[:, 1])
            concept_correct += concept_pred_labels.eq(concept_labels).sum().item()
            total_concepts += concept_labels.numel()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    concept_accuracy = 100. * concept_correct / total_concepts

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%), Concept Accuracy: {concept_accuracy:.2f}%\n')

