import types
import torchvision.models as models
from .utils import model_urls, load_pretrained


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


input_sizes = {}
means = {}
stds = {}
for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]


pretrained_settings = {}
for model_name in __all__:
    pretrained_settings[model_name] = {
        'imagenet': {
            'url': model_urls[model_name],
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name],
            'num_classes': 1000
        }
    }


def modify_resnets(model):
    # Modify attributs
    model.last_linear = model.fc
    model.fc = None

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def resnet18(num_classes: int = 1000, pretrained: str = 'imagenet', requires_grad: bool = True):
    """Constructs a ResNet-18 model.
    """
    model = models.resnet18(pretrained=False)
    if pretrained is not None:
        if pretrained == 'imagenet':
            settings = pretrained_settings['resnet18'][pretrained]
            model = load_pretrained(model, num_classes, settings)
        else:
            raise ValueError(
                f"Unknown pretrain name: {pretrained}. "
                f"Should be 'imagenet' or None"
            )
    model = modify_resnets(model)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def resnet34(num_classes: int = 1000, pretrained: str = 'imagenet', requires_grad: bool = True):
    """Constructs a ResNet-34 model.
    """
    model = models.resnet34(pretrained=False)
    if pretrained is not None:
        if pretrained == 'imagenet':
            settings = pretrained_settings['resnet34'][pretrained]
            model = load_pretrained(model, num_classes, settings)
        else:
            raise ValueError(
                f"Unknown pretrain name: {pretrained}. "
                f"Should be 'imagenet' or None"
            )
    model = modify_resnets(model)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def resnet50(num_classes: int = 1000, pretrained: str = 'imagenet', requires_grad: bool = True):
    """Constructs a ResNet-50 model.
    """
    model = models.resnet50(pretrained=False)
    if pretrained is not None:
        if pretrained == 'imagenet':
            settings = pretrained_settings['resnet50'][pretrained]
            model = load_pretrained(model, num_classes, settings)
        else:
            raise ValueError(
                f"Unknown pretrain name: {pretrained}. "
                f"Should be 'imagenet' or None"
            )
    model = modify_resnets(model)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def resnet101(num_classes: int = 1000, pretrained: str = 'imagenet', requires_grad: bool = True):
    """Constructs a ResNet-101 model.
    """
    model = models.resnet101(pretrained=False)
    if pretrained is not None:
        if pretrained == 'imagenet':
            settings = pretrained_settings['resnet101'][pretrained]
            model = load_pretrained(model, num_classes, settings)
        else:
            raise ValueError(
                f"Unknown pretrain name: {pretrained}. "
                f"Should be 'imagenet' or None"
            )
    model = modify_resnets(model)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def resnet152(num_classes: int = 1000, pretrained: str = 'imagenet', requires_grad: bool = True):
    """Constructs a ResNet-152 model.
    """
    model = models.resnet152(pretrained=False)
    if pretrained is not None:
        if pretrained == 'imagenet':
            settings = pretrained_settings['resnet152'][pretrained]
            model = load_pretrained(model, num_classes, settings)
        else:
            raise ValueError(
                f"Unknown pretrain name: {pretrained}. "
                f"Should be 'imagenet' or None"
            )
    model = modify_resnets(model)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
