

VALID_DATASET = ["iPER", "iPER_ICCV", "MotionSynthetic"]


def create_dataset_protocols(dataset, data_dir):
    assert dataset in VALID_DATASET

    if dataset == "iPER":
        from .iPER import IPERProtocol
        return IPERProtocol(data_dir)

    elif dataset == "iPER_ICCV":
        from .iPER import ICCVIPERProtocol
        return ICCVIPERProtocol(data_dir)

    elif dataset == "MotionSynthetic":
        from .MotionSynthetic import MotionSyntheticProtocol
        return MotionSyntheticProtocol(data_dir)

    else:
        raise ValueError("{} must be in {}".format(dataset, VALID_DATASET))
