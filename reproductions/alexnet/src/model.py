"""AlexNet adapted for CIFAR-10 with 32x32 input.

The original AlexNet expects 227x227 input. Its first layer uses an 11x11 kernel
with stride 4, which would reduce a 32x32 image to 6x6 and make subsequent
pooling impractical. This implementation keeps the "5 conv + 3 fc" structure and
channel progression of AlexNet, while adjusting kernel sizes and strides for
32x32 input.

Feature map shape:
  32 -> Conv5x5(pad2) -> 32 -> Pool3/s2 -> 15
     -> Conv5x5(pad2) -> 15 -> Pool3/s2 -> 7
     -> Conv3x3 -> 7 -> Conv3x3 -> 7 -> Conv3x3 -> 7 -> Pool3/s2 -> 3
Flattened size: 256 * 3 * 3 = 2304, followed by FC6(4096)-FC7(4096)-FC8(10).
"""

import torch.nn as nn


class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: original 11x11/96/stride4 -> 5x5/64/stride1
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # 32 -> 15

            # Conv2: original 5x5/256 -> 5x5/192
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # 15 -> 7

            # Conv3: 3x3/384, no pooling
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 3x3/256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 3x3/256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # 7 -> 3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick shape check: python -m src.model
    import torch
    m = AlexNetCIFAR()
    out = m(torch.randn(2, 3, 32, 32))
    print("Output shape:", out.shape)
    print("Parameters: %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))