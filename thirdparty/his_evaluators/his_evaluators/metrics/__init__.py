from .metrics import BaseMetric, PerceptualMetric, SSIMMetric, PSNRMetric, \
    InceptionScoreMetric, FIDMetric, FreIDMetric, ReIDScore, ScaleShapePoseError


TYPES = [
    "ssim",
    "psnr",
    "lps",
    "is",
    "fid",
    "OS-CS-reid",
    "OS-freid",
    "PCB-CS-reid",
    "PCB-freid",
    "SSPE",
    "face-CS",
    "face-FD"
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
    "PCB-freid": "pcb_feats",
    "SSPE": "SSPE",
    "face-CS": "face-CS",
    "face-FD": "face_feats"
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
    "PCB-freid": BaseMetric.LOWER,
    "SSPE": BaseMetric.LOWER,
    "face-CS": BaseMetric.HIGHER,
    "face-FD": BaseMetric.LOWER
}

METRIC_DICT = dict()


def register_metrics(types, device, has_detector=True):
    global TYPES, METRIC_DICT

    metric_dict = dict()

    for name in types:
        assert name in TYPES

        if name in METRIC_DICT:
            metric_dict[name] = METRIC_DICT[name]
            continue

        if name == "ssim":
            metric_dict[name] = SSIMMetric()
        elif name == "psnr":
            metric_dict[name] = PSNRMetric()
        elif name == "lps":
            metric_dict[name] = PerceptualMetric(device)
        elif name == "is":
            metric_dict[name] = InceptionScoreMetric(device)
        elif name == "fid":
            metric_dict[name] = FIDMetric(device)
        elif name == "OS-CS-reid":
            metric_dict[name] = ReIDScore(device, reid_name=BaseMetric.OSreID, has_detector=has_detector)
        elif name == "OS-freid":
            metric_dict[name] = FreIDMetric(device, reid_name=BaseMetric.OSreID, has_detector=has_detector)
        elif name == "PCB-CS-reid":
            metric_dict[name] = ReIDScore(device, reid_name=BaseMetric.PCBreID, has_detector=has_detector)
        elif name == "PCB-freid":
            metric_dict[name] = FreIDMetric(device, reid_name=BaseMetric.PCBreID, has_detector=has_detector)
        elif name == "SSPE":
            metric_dict[name] = ScaleShapePoseError(device)
        elif name == "face-CS":
            from .metrics import FaceSimilarityScore
            metric_dict[name] = FaceSimilarityScore(device=device, has_detector=has_detector)
        elif name == "face-FD":
            from .metrics import FaceFrechetDistance
            metric_dict[name] = FaceFrechetDistance(device=device, has_detector=has_detector)
        else:
            raise ValueError(name)

        METRIC_DICT[name] = metric_dict[name]

    return metric_dict
