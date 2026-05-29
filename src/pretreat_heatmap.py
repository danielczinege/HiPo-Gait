import re
import os
import argparse
import time
import warnings
import cv2 as cv
import numpy as np
import pickle
import tqdm
from multiprocessing import Pool
from utils import list_dir_recursively

# General variables
casiab_fn_fmt_re = re.compile('([0-9]{3})-(nm|cl|bg)-([0-9]{2})-([0-9]{3})')
tum_gaid_fn_fmt_re = re.compile('(p[0-9]{3})-((n|b|s)[0-9]{2})')
fvgb_fn_fmt_re = re.compile('(session[1-3])/([0-9]{3})/([0-9]{2})')
TERM_FN = '.npy'

# Useful functions
sort_dict = lambda d: dict(sorted(d.items()))

HM_GROUPS = {
    'limb-based': {
        'left-arm': [0, 1, 3, 5, 7, 9],
        'left-leg': [11, 13, 15],
        'right-arm': [2, 4, 6, 8, 10],
        'right-leg': [12, 14, 16]
    },
    'full-body': {
        'full-body': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    },
    'limb-based_wo_head': {
        'left-arm': [5, 7, 9],
        'left-leg': [11, 13, 15],
        'right-arm': [6, 8, 10],
        'right-leg': [12, 14, 16]
    }
}


def fvgb_get_view_variation(session, seq):
    """
        Retrieve the angle and the sequence of every sequence
        for the FVG-B dataset.

        Return
        ------
        view. str

        Contains one of the following values
        -45, 000 or +45

        var. str

        Variation for the video sequence:
        NM: Normal Walking
        WSL: Walking speed (slow speed)
        WSF: Walking speed (fast speed)
        CB: Carrying a bag
        CL: Changing clothes
    """
    _VIEWS = ['-45', '000', '+45']
    seq = int(seq)

    # Retrieve view:
    ## seq are defined in [1 - 12], so need to be shifted to [0 - 11]
    ##  to use modulus operator correctly.
    view = _VIEWS[(seq - 1) % 3]

    # Retrieve variations
    if session == 'session1':
        if seq in (1, 2, 3):
            var = 'NM'  # Normal walking
        elif seq in (4, 5, 6):
            var = 'WSL'  # Walking Speed (Slow)
        elif seq in (7, 8, 9):
            var = 'WSF'  # Walking speed (Fast)
        else:
            var = 'CB'  # Carrying a bag or a hat
    else:
        if seq in (1, 2, 3):
            var = 'NM'  # Normal walking
        elif seq in (4, 5, 6):
            var = 'WSL'  # Walking Speed (Slow)
        elif seq in (7, 8, 9):
            var = 'CL'  # Changing clothes
        else:
            var = 'MP'  # Multiple person

    return view, var

def fuse_heatmap_channels(hm: np.array, groups: list, agg_ope='gauss') -> np.array:

    r"""Fuses the heatmap joint channels from a heatmap array into a single
        heatmap containing the whole body skeletton. Fusion is performed by assuming each heatmap
        channel holds a gaussian distribution with mean 0 and std 1.

        Arguments
        ---------
        hm: Array
            Separate channel heatmaps to be fused

        groups: List
            List of lists of joint indexes to be grouped

        agg_ope: str ('gauss' or 'max')
            Aggregation operation for joint heatmaps groupping
            Allowed values: gauss or max


        Agg. operations
        ---------------

        - gauss: Each heatmap is assumed in a gaussian distribution centered
            at 0 with std 1. Thus, suming for each heatmap group is performed
            by element-wise sum of the joint heatmaps, and dividing the heatmap
            group by the square root of the number of joint grouped:

          ```
            fus_hm{i} =  \frac{\sum_{j \in J_g} hm_{j}} {\sqrt{n_{g}}} \forall_{i} \in G
          ```
          
          where $hm_{j}$ is the raw heatmap for joint j, $J_g$ is the set of
          joint indexes included in the heatmap group, and $n_{g}$ is the number
          of joints grouped. $i$ refers to the i-th heatmap group, while G
          is the set of groups

        - max: Heatmaps are grouped by maximizing the values among all joint
            heatmaps in the same position

        Return
        ------
        Array containing the heatmaps with the aggrupation
    """

    n_c, h, w = hm.shape
    n_groups = len(groups)

    fus_hm = np.zeros((n_groups, h, w), dtype=hm.dtype)

    if agg_ope == 'gauss':
        for i, group in enumerate(groups):
            fus_hm[i] = hm[group].sum(axis=0, keepdims=True) / np.sqrt(len(group))
    elif agg_ope == 'max':
        for i, group in enumerate(groups):
            fus_hm[i] = hm[group].max(axis=0, keepdims=True)
    else:
        raise ValueError(f'{agg_ope} operation not supported')

    return fus_hm


def expand_to_square_size(img: np.array) -> np.array:
    """
        Expands the image width or height to convert into square size

        Arguments
        ---------
        img: Array
            Image to be expanded

        Returns
        -------
        Array: Expanded image
    """

    height, width = img.shape[-2:]
    channels = img.shape[0] if len(img.shape) == 3 else 1

    # No expansion is required
    if height == width:
        return img

    # Expansion required on the width or height
    max_dim = max(height, width)

    exp_img = np.zeros((channels, max_dim, max_dim), dtype=img.dtype)

    if width < max_dim:
        pad = max_dim - width
        half_pad = int(pad // 2)
        exp_img[:, :, half_pad: width + half_pad] = img
    else:
        pad = max_dim - height
        half_pad = int(pad // 2)
        exp_img[:, half_pad: height + half_pad, :] = img

    if channels == 1:
        exp_img = exp_img[0]

    return exp_img


def process_video(args):
    (dataset_name, save_results, heatmaps_dir, height_out,
     agg_method, hm_group, replace_if_exists, frame_num_re_str,
     subdir, fname) = args

    # Frame regex finder
    frame_num_re = re.compile(frame_num_re_str)

    # Retrieve video subdirs
    hm_dir = os.path.join(heatmaps_dir, subdir, fname)

    # Parse subject id (sid), seq. type, and view following dataset struct.
    if dataset_name == 'CASIAB':
        match = casiab_fn_fmt_re.search(hm_dir)

        if not match:
            print(f'Found no regular video {fname}. Ignored')
            return

        sid, seq, view = (match.group(1), match.group(2) + '-' + match.group(3), match.group(4))
    elif dataset_name == 'CCPG':
        sid, seq = subdir.split('/')[-2:]
        view = fname.replace('_hm', '')

    elif dataset_name == 'FVG-B':
        sess, sid = subdir.split('/')[:2]
        seq_num = fname[:2]
        view, _ = fvgb_get_view_variation(sess, int(seq_num))

        # Restructure seq name
        seq = f'{sess}-{seq_num}'
    elif dataset_name == 'SUSTech1K':
        sid, seq, view = subdir.split('/')[-3:]

    else:
        print(f'Dataset {dataset_name} not supported yet')
        return

    vid_fname = f'{sid}-{seq}-{view}'

    # Create the subdirs sid/seq/view
    subdir_path = os.path.join(save_results, sid, seq, view)
    os.makedirs(subdir_path, exist_ok=True)

    # Discard already processed heatmaps
    if not replace_if_exists and os.path.isfile(os.path.join(subdir_path, f'{view}.pkl')):
        return

    # List and sort all the raw heatmap frames within heatmap dir according to frame number
    frame_nums = dict()
    for raw_hm_fn in os.listdir(hm_dir):

        if not raw_hm_fn.endswith(TERM_FN):
            warnings.warn(f'Found non-valid numpy file {raw_hm_fn}. Skipping')
            continue

        ## Retrieve frame number from raw heatmap filename
        found_frame = frame_num_re.search(raw_hm_fn)
        try:
            frame_num = int(found_frame.group(1))
        except:
            warnings.warn(f'Cannot retrieve frame number for heatmap file {raw_hm_fn}. Skipping!!')
            continue

        if frame_num in frame_nums:
            warnings.warn(f'Found duplicated frame {frame_num} in video {vid_fname}. Skipping !!')
            continue
        else:
            frame_nums[frame_num] = os.path.join(hm_dir, raw_hm_fn)

    frame_nums = sort_dict(frame_nums)

    if not frame_nums:
        warnings.warn('"{}" does not contains any frame and will be discarded'.format(vid_fname))
        return

    # Process all heatmap frames following body part rep. and store as single array
    hm_proces = []

    for frame_num, raw_hm_fn in frame_nums.items():
        raw_hm_data = np.load(raw_hm_fn)

        # Fuse joint heatmaps following body part representation
        if agg_method == 'sum':  # gauss sum
            raw_hm_data = fuse_heatmap_channels(raw_hm_data, groups=hm_group, agg_ope='gauss')
        elif agg_method == 'max':
            raw_hm_data = fuse_heatmap_channels(raw_hm_data, groups=hm_group, agg_ope='max')

        # Resize hm to the output dim file
        if raw_hm_data.shape[-2:] != (height_out, height_out):
            raw_hm_data = expand_to_square_size(raw_hm_data)
            ## CV requires aray dimension as [height_out, height_out, groups]
            if len(raw_hm_data.shape) == 3:
                raw_hm_data = np.moveaxis(raw_hm_data, (0, 1, 2), (2, 0, 1))

            raw_hm_data = cv.resize(raw_hm_data, (height_out, height_out))

            if len(raw_hm_data.shape) == 3:
                raw_hm_data = np.moveaxis(raw_hm_data, (2, 0, 1), (0, 1, 2))

        hm_proces.append(raw_hm_data)

    hm_proces = np.array(hm_proces)

    # Save array into file
    with open(os.path.join(subdir_path, f'{view}.pkl'), 'wb') as out_file:
        pickle.dump(hm_proces, out_file)

def main(args):

    # Collect input arguments
    heatmaps_dir = args.input_path
    save_results = args.output_path
    height_out = args.height_output
    dataset_name = args.dataset
    replace_if_exists = args.replace_if_exists
    frame_format = args.frame_format
    agg_method = args.agg_method
    grouping = args.grouping
    num_workers = args.num_workers

    # Check if output results directory exists or create it
    if not os.path.isdir(save_results):
        try:
            os.mkdir(save_results)
        except Exception as e:
            print('Failed to create output directory {}'.format(str(e)))
            exit(-1)

    # Retrieve heatmap grouping
    try:
        hm_grouping = HM_GROUPS[grouping]
    except KeyError as e:
        print('Failed to retrieve heatmap grouping {}: {}'.format(grouping, str(e)))
        exit(-1)

    # Retrieve list of video heatmaps dir to scan
    print(f'Scanning {heatmaps_dir}')
    dirlist = list_dir_recursively(heatmaps_dir, keep_only_dir=True)

    hm_grouping_list = list(hm_grouping.values())

    # Create arguments for pool
    pool_args = [
        (dataset_name, save_results, heatmaps_dir, height_out,
         agg_method, hm_grouping_list, replace_if_exists, frame_format,
         subdir, fname)
        for subdir, fname in dirlist
    ]

    if num_workers == 1:
        for args_item in tqdm.tqdm(pool_args, desc='Videos processed'):
            process_video(args_item)
    else:
        try:
            with Pool(num_workers) as pool:
                for _ in tqdm.tqdm(pool.imap_unordered(process_video, pool_args), total=len(pool_args), desc='Videos processed'):
                    pass
        except KeyboardInterrupt:
            print('Interrupt signal received. Closing all the processes')
        except Exception as e:
            print('An exception ocurred: ', str(e))

if __name__ == '__main__':

    ### Input Arguments
    parser = argparse.ArgumentParser(description='Preprocess a set of raw pose heatmaps into serialized pickle format')

    parser.add_argument('-i', '--input_path', help='Directory path containing the raw pose heatmaps', type=str, required=True)
    parser.add_argument('-o', '--output_path', help='Directory path where the processed heatmaps image files will be placed', type=str, required=True)
    parser.add_argument('--height_output', help='Output frame height', type=int, required=False, default=64)
    parser.add_argument('-d', '--dataset', default='CASIAB', type=str, help='Dataset for pretreatment.', choices=['CASIAB', 'CCPG', 'FVG-B', 'SUSTech1K'])
    parser.add_argument('-g', '--grouping', help='Body part grouping for heatmap fusion', type=str, default='limb-based', choices=['limb-based', 'full-body', 'limb-based_wo_head'])
    parser.add_argument('--replace_if_exists', action='store_true', help='Replace file if exists', default=False)
    parser.add_argument('--frame_format', help='Frame number format followed in the heatmap filenames', type=str, default='frame-([0-9]+)')
    parser.add_argument('--agg_method', help='Heatmap aggregation function', type=str, default='sum', choices=['sum', 'max'])
    parser.add_argument('--num_workers', help='Number of workers to process the dataset', type=int, default=1)

    args = parser.parse_args()
    
    # Run program
    main(args)
