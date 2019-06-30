import torch
import torch.nn.functional as F
from networks.losses.cx_loss import symetric_CX_loss


class TVLoss(torch.nn.Module):
    """
    TV loss
    """

    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


class CXReconLoss(torch.nn.Module):

    """
    contexutal loss with vgg network
    """

    def __init__(self, feat_extractor, device=None, weight=1):
        super(CXReconLoss, self).__init__()
        self.feat_extractor = feat_extractor
        self.device = device
        if device is not None:
            self.feat_extractor = self.feat_extractor.to(device)
        # self.feat_extractor = self.feat_extractor.cuda()
        self.weight = weight

    def forward(self, imgs, recon_imgs, coarse_imgs=None):
        if self.device is not None:
            imgs = imgs.to(self.device)
            recon_imgs = recon_imgs.to(self.device)
            if coarse_imgs is not None:
                coarse_imgs = coarse_imgs.to(self.device)

        imgs = F.interpolate(imgs, (224,224))
        recon_imgs = F.interpolate(recon_imgs, (224,224))

        ori_feats, _ = self.feat_extractor(imgs)
        recon_feats, _ = self.feat_extractor(recon_imgs)
        if coarse_imgs is not None:
            coarse_imgs = F.interpolate(coarse_imgs, (224,224))
            coarse_feats, _ =self.feat_extractor(coarse_imgs)
            return self.weight * (symetric_CX_loss(ori_feats, recon_feats) )
        return self.weight * symetric_CX_loss(ori_feats, recon_feats)


class MaskDisLoss(torch.nn.Module):
    """
    The loss for mask discriminator
    """
    def __init__(self, weight=1):
        super(MaskDisLoss, self).__init__()
        self.weight = weight
        self.leakyrelu = torch.nn.LeakyReLU()

    def forward(self, pos, neg):
        # return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(self.leakyrelu(1.-pos)) + torch.mean(self.leakyrelu(1.+neg)))


class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        # return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(F.relu(1.-pos)) + torch.mean(F.relu(1.+neg)))


class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)


class L1ReconLoss(torch.nn.Module):
    """
    L1 Reconstruction loss for two imgae
    """
    def __init__(self, weight=1):
        super(L1ReconLoss, self).__init__()
        self.weight = weight

    def forward(self, imgs, recon_imgs, masks=None):
        if masks is None:
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs))
        else:
            #print(masks.view(masks.size(0), -1).mean(1).size(), imgs.size())
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs) / masks.view(masks.size(0), -1).mean(1).view(-1,1,1,1))


# CX
# [0,9,13,17]
class PerceptualLoss(torch.nn.Module):
    """
    Use vgg or inception for perceptual loss, compute the feature distance, (todo)
    """
    def __init__(self, weight=1, layers=(0, 9, 13, 17), feat_extractors=None):
        super(PerceptualLoss, self).__init__()
        self.weight = weight
        self.feat_extractors = feat_extractors
        self.layers = layers

    def forward(self, imgs, recon_imgs):
        imgs = F.interpolate(imgs, (224,224))
        recon_imgs = F.interpolate(recon_imgs, (224,224))
        feats = self.feat_extractors(imgs, self.layers)
        recon_feats = self.feat_extractors(recon_imgs, self.layers)
        loss = 0
        for feat, recon_feat in zip(feats, recon_feats):
            loss = loss + torch.mean(torch.abs(feat - recon_feat))
        return self.weight*loss


class StyleLoss(torch.nn.Module):
    """
    Use vgg or inception for style loss, compute the feature distance, (todo)
    """
    def __init__(self, weight=1, layers=[0,9,13,17], feat_extractors=None):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.feat_extractors = feat_extractors
        self.layers = layers

    def gram(self, x):
        gram_x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        return torch.bmm(gram_x, torch.transpose(gram_x, 1, 2))

    def forward(self, imgs, recon_imgs):
        imgs = F.interpolate(imgs, (224,224))
        recon_imgs = F.interpolate(recon_imgs, (224,224))
        feats = self.feat_extractors(imgs, self.layers)
        recon_feats = self.feat_extractors(recon_imgs, self.layers)
        loss = 0
        for feat, recon_feat in zip(feats, recon_feats):
            loss = loss + torch.mean(torch.abs(self.gram(feat) - self.gram(recon_feat))) / (feat.size(2) * feat.size(3) )
        return self.weight*loss


class ReconLoss(torch.nn.Module):
    """
    Reconstruction loss contain l1 loss, may contain perceptual loss

    """
    def __init__(self, chole_alpha, cunhole_alpha, rhole_alpha, runhole_alpha):
        super(ReconLoss, self).__init__()
        self.chole_alpha = chole_alpha
        self.cunhole_alpha = cunhole_alpha
        self.rhole_alpha = rhole_alpha
        self.runhole_alpha = runhole_alpha

    def forward(self, imgs, coarse_imgs, recon_imgs, masks):
        masks_viewed = masks.view(masks.size(0), -1)
        return self.rhole_alpha*torch.mean(torch.abs(imgs - recon_imgs) * masks / masks_viewed.mean(1).view(-1,1,1,1))  + \
                self.runhole_alpha*torch.mean(torch.abs(imgs - recon_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))  + \
                self.chole_alpha*torch.mean(torch.abs(imgs - coarse_imgs) * masks / masks_viewed.mean(1).view(-1,1,1,1))   + \
                self.cunhole_alpha*torch.mean(torch.abs(imgs - coarse_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))
