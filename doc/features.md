# Version 1.x

## ImPer-1.0
The basic features of Impersonator, including:

* `Motion Imitation`: [demo_imitator.py](../demo_imitator.py), [Imitator.py](../models/imitator.py);
* `Appearance Transfer`: [demo_swap.py](../demo_swap.py), [Swapper.py](../models/swapper.py);
* `Novel View Synthesis`. [demo_view.py](../demo_view.py), [Viewer.py](../models/viewer.py);

## ImPer-1.1
Fix some bugs and update the `post_personalize` function of each task, replacing the `pseudo_masks` with the 
`tsf_inputs[:, -1:] (src_inputs[:, -1:])`.