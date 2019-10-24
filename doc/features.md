# Version 1.x

## ImPer-1.0
The basic features of Impersonator, including:

* `Motion Imitation`: [demo_imitator.py](../run_imitator.py), [Imitator.py](../models/imitator.py);
* `Appearance Transfer`: [demo_swap.py](../run_swap.py), [Swapper.py](../models/swapper.py);
* `Novel View Synthesis`. [demo_view.py](../run_view.py), [Viewer.py](../models/viewer.py);

## ImPer-1.1
Fix some bugs and update the `post_personalize` function of each task, replacing the `pseudo_masks` with the 
`tsf_inputs[:, -1:] (src_inputs[:, -1:])`.

## ImPer-1.2

### ImPer-1.2.1
- [x] 09/27/2019, merge running scripts and demo scripts.

- [x] 10/05/2019, optimize the minimal requirements of GPU memory (at least `3.8GB` available).

### Imper-1.2.2
- [x] 10/24/2019, add the training document [train.md](../doc/train.md).
