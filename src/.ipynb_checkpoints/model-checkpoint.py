import torch.nn as nn
import torch.nn.functional as F

class ConceptBottleneckCNN(nn.Module):
    def __init__(self, num_concepts):
        super(ConceptBottleneckCNN, self).__init__()
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        # Concept bottleneck layer
        self.fc_concepts = nn.Linear(320, num_concepts)
        # Classifier layer
        self.fc_classifier = nn.Linear(num_concepts, 10)

    def forward(self, x):
        # Feature extraction
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        # Predict concepts
        concepts = torch.sigmoid(self.fc_concepts(x))
        # Predict class using concepts
        logits = self.fc_classifier(concepts)
        return concepts, logits
