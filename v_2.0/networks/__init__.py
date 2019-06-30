from .bodymesh import HumanModelRecovery, SMPLRendererTrainer, SMPLRenderer, SMPLRendererV2
from .utils.detectors import PersonMaskRCNNDetector


def create_by_name(network_name, *args, **kwargs):
    """
    :param network_name:
    :param args:
    :param kwargs:
    :return:
    """

    if network_name == 'res_unet':
        from networks.generators import ImpersonatorGenerator
        network = ImpersonatorGenerator(*args, **kwargs)

    elif network_name == 'res_unet_front':
        from networks.generators import ImpersonatorFrontGenerator
        network = ImpersonatorFrontGenerator(*args, **kwargs)

    elif network_name == 'res_plain_unet':
        from networks.generators import ImpersonatorPlainGenerator
        network = ImpersonatorPlainGenerator(*args, **kwargs)

    elif network_name == 'res_plain_unet_front':
        from networks.generators import ImpersonatorPlainFrontGenerator
        network = ImpersonatorPlainFrontGenerator(*args, **kwargs)

    elif network_name == 'res_share_unet':
        from networks.generators import ImperShareDecoderGenerator
        network = ImperShareDecoderGenerator(*args, **kwargs)

    elif network_name == 'concat':
        from networks.generators import ConcatOrTextureWarpingGenerator
        network = ConcatOrTextureWarpingGenerator(*args, **kwargs)

    elif network_name == 'texture':
        from networks.generators import ConcatOrTextureWarpingGenerator
        network = ConcatOrTextureWarpingGenerator(*args, **kwargs)

    elif network_name == 'feature':
        from networks.generators import FeatureWarpingGenerator
        network = FeatureWarpingGenerator(*args, **kwargs)

    elif network_name == 'gate_unet':
        from networks.generators import GateUnetGenerator
        network = GateUnetGenerator(*args, **kwargs)

    elif network_name == 'isag':
        from networks.generators import InpaintSANet
        network = InpaintSANet(*args, **kwargs)

    elif network_name == 'isad':
        from networks.discriminators import InpaintSADirciminator
        network = InpaintSADirciminator(*args, **kwargs)

    elif network_name == 'snd':
        from networks.discriminators import SNDiscriminator
        network = SNDiscriminator(*args, **kwargs)

    elif network_name == 'patch':
        from networks.discriminators.discriminator import PatchDiscriminator
        network = PatchDiscriminator(*args, **kwargs)

    elif network_name == 'global_local':
        from networks.discriminators.discriminator import GlobalLocalPatchDiscriminator
        network = GlobalLocalPatchDiscriminator(*args, **kwargs)

    else:
        raise ValueError("Network %s not recognized." % network_name)

    print("Network %s was created" % network_name)

    return network


def init_weights(net, init_type='normal', gain=0.02):
    from torch.nn import init

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


if __name__ == '__main__':
    import ipdb
    import torch
    net_isag = create_by_name(network_name='isag')
    net_isad = create_by_name(network_name='isad', c_dim=3)

    images = torch.rand(2, 3, 256, 256)
    masks = torch.rand(2, 1, 256, 256)
    coarse_imgs, fake_imgs = net_isag(images, masks)
    print(coarse_imgs.shape, fake_imgs.shape)

    out = net_isad(fake_imgs)
    print(out.shape)

    ipdb.set_trace()
