import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 다운샘플 경로(채널/스트라이드 불일치 보정)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # 스킵 연결 입력 보존

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.inplanes = 64  # 현재 스테이지 입력 채널 수 추적
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 글로벌 평균 풀링
        self.fc = nn.Linear(
            512 * block.expansion, num_classes
        )  # 최종 특징 2048(=512*4)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  # 다운샘플 경로 구성 필요 여부 판단
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet50(num_classes: int = 2, in_channels: int = 3) -> nn.Module:
    """
    ResNet-50 (Bottleneck 기반) 생성 함수
    - 블록 구성: [3, 4, 6, 3]
    - 최종 특징 차원: 2048 (= 512 × expansion 4)
    - 다운샘플 경로: 스트라이드/채널 불일치 시 1×1 Conv + BN 사용
    """
    return ResNet(
        Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels
    )
