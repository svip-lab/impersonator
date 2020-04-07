

VALID_DATASET = ["iPER", "MotionSynthetic", "Youtube-Dancer-8"]


def create_dataset_protocols(dataset, data_dir):
    assert dataset in VALID_DATASET

    if dataset == "iPER":
        from .iPER import IPERProtocol
        return IPERProtocol(data_dir)

    else:
        raise ValueError("{} must be in {}".format(dataset, VALID_DATASET))
