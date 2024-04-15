from torch import nn



def resnet10t_instance_norm():
    from timm.models.resnet import resnet10t

    model = resnet10t(
        in_chans=3,
    )
    model.fc = nn.Identity()
    return nn.Sequential(nn.InstanceNorm2d(3), model)

def resnet32_instance_norm():
    from medAI.modeling.resnet import resnet32

    model = resnet32()
    model.fc = nn.Identity()
    return nn.Sequential(nn.InstanceNorm2d(3), model)