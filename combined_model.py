import torch
import torch.nn as nn

class EfficientSRCNN(nn.Module):
    def __init__(self):
        super(EfficientSRCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)


class CombinedTrainableSRModel(nn.Module):
    def __init__(self, checkpoint_paths):
        super(CombinedTrainableSRModel, self).__init__()
        self.models = nn.ModuleList()
        for path in checkpoint_paths:
            model = EfficientSRCNN()
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            self.models.append(model)

        # combiner layer
        self.combiner = nn.Conv2d(3 * len(self.models), 3, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            outputs = [model(x) for model in self.models]
        combined = torch.cat(outputs, dim=1)  # concatenează pe axa canalelor (C)
        return torch.clamp(self.combiner(combined), 0.0, 1.0)
