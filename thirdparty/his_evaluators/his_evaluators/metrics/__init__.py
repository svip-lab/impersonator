from .metrics import BaseMetric, PerceptualMetric, SSIMMetric, PSNRMetric, \
    InceptionScoreMetric, FIDMetric, FreIDMetric, ReIDScore


TYPES = [
    "ssim",
    "psnr",
    "lps",
    "is",
    "fid",
    "OS-CS-reid",
    "OS-freid",
    "PCB-CS-reid",
    "PCB-freid"
]

TYPES_RESULTS_MAP = {
    "ssim": "ssim",
    "psnr": "psnr",
    "lps": "lps",
    "is": "inception_feats",
    "fid": "inception_feats",
    "OS-CS-reid": "osnet_feats",
    "OS-freid": "osnet_feats",
    "PCB-CS-reid": "pcb_feats",
    "PCB-freid": "pcb_feats"
}

TYPES_QUALITIES = {
    "ssim": BaseMetric.LOWER,
    "psnr": BaseMetric.HIGHER,
    "lps": BaseMetric.LOWER,
    "is": BaseMetric.HIGHER,
    "fid": BaseMetric.LOWER,
    "OS-CS-reid": BaseMetric.HIGHER,
    "OS-freid": BaseMetric.LOWER,
    "PCB-CS-reid": BaseMetric.HIGHER,
    "PCB-freid": BaseMetric.LOWER
}


def register_metrics(types, device):
    global TYPES

    metric_dict = dict()

    for name in types:
        assert name in TYPES

        if name == 'ssim':
            metric_dict[name] = SSIMMetric()
        elif name == "psnr":
            metric_dict[name] = PSNRMetric()
        elif name == 'lps':
            metric_dict[name] = PerceptualMetric(device)
        elif name == 'is':
            metric_dict[name] = InceptionScoreMetric(device)
        elif name == 'fid':
            metric_dict[name] = FIDMetric(device)
        elif name == 'OS-CS-reid':
            metric_dict[name] = ReIDScore(device, reid_name=BaseMetric.OSreID)
        elif name == 'OS-freid':
            metric_dict[name] = FreIDMetric(device, reid_name=BaseMetric.OSreID)
        elif name == 'PCB-CS-reid':
            metric_dict[name] = ReIDScore(device, reid_name=BaseMetric.PCBreID)
        elif name == 'PCB-freid':
            metric_dict[name] = FreIDMetric(device, reid_name=BaseMetric.PCBreID)
        else:
            raise ValueError(name)

    return metric_dict
