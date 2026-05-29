## Pose heatmap pretreatment

Following instructions indicate the procedure to pretreat the raw pose heatmaps estimated from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) in serialized pickle format accepted by the scripts.

Raw heatmaps should be structured depending on the dataset. See examples below:

### CASIAB dataset

```
    HEATMAP ROOT PATH/
        video #1 / (a subdir for each video containing all frames. e.g., 001-nm-01-090, 005-cl-02-018, etc)
            frame-0.npy (numpy array with size [17, H, W])
            frame-1.npy
            frame-2.npy
            ......

        video #2 /
            frame-0.npy
            frame-1.npy
            frame-2.npy
            ......

        ......
```

### FVG-B dataset

```
    HEATMAP ROOT PATH/
        session number (1, 2, or 3)/
            subject #1/
                sequence #1 (01, 02, etc)/
                    frame-0.npy
                    frame-1.npy
                    frame-2.npy
                    ......
                sequence #2/
                    frame-0.npy
                    frame-1.npy
                    frame-2.npy
                    ......
                ......
            ......
        ......
```

### CCPG dataset

```
    HEATMAP ROOT PATH/
        subject #1/
            sequence type #1 (U0_D0, U1_D1, etc)/
                sequence number #1/
                    frame-0.npy
                    frame-1.npy
                    frame-2.npy
                    ......
                sequence number #2 (01_0, 02_0, etc)/
                    frame-0.npy
                    frame-1.npy
                    frame-2.npy
                    ......
                ......
            ......
        ......
```

### SUSTech1K dataset

```
    HEATMAP ROOT PATH/
        subject #1/
            sequence type #1 (00-nm, 01-cr, etc.)/
                view (000, 000-far, 045, etc)/
                    frame-0.npy
                    frame-1.npy
                    frame-2.npy
                    ......
                view #2/
                    frame-0.npy
                    frame-1.npy
                    frame-2.npy
                    ......
                ......
            ......
        ......
```

Note that each `.npy` file contains a numpy array with 17 channels (one for each skeleton joint).

Run script `src/pretreat_heatmaps.py` as follows:

```bash
python3 src/pretreat_heatmap.py \
    -i <heatmap root path> \
    -o <out dir for preprocessed hms> \
    -d CASIAB \ # or CCPG, or FVG-B, or SUSTech1K \
    --num_workers <number of workers to parallelize pretreatment>
```

- Check `-h` option for more details.
- Option `-g` enables the selection of different body joint heatmap groupings:
  - *limb-based* (default): Limb-based heatmap representation presented in manuscript.
  - *full-body*: Heatmap representation containing all the joints of human body.
  - *limb-based_wo_head*: Limb-based heatmap representation without including the head joints.


Pretreated heatmaps will be stored in the following directory structure:

```
    DATASET_ROOT/
        subject id/
            seq type/
                    view/
                        view.pkl (contains all frames)
                ......
            ......
        ......
```