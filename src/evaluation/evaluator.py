import os
import numpy as np
from sklearn.metrics import roc_curve
from utils import get_msg_mgr

from .metric import cuda_dist, compute_ACC_mAP, evaluate_many

def de_diag(acc, each_angle=False):
    # Exclude identical-view cases
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result

def make_bin_labels(probe_y, gallery_y):
    # Ensure both probe_y and gallery_y can be broadcasted together
    probe_y = np.expand_dims(probe_y, 1)                # dims: [len(probe_y), 1]
    gallery_y = np.expand_dims(gallery_y, 0)            # dims: [1, len(gallery_y)]

    # Compare gallery label against each probe label by parallelization
    bin_labels = (gallery_y == probe_y)

    return bin_labels.astype(np.uint8) #bin_labels.byte()

def cross_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    '''More details can be found: More details can be found in 
        [A Comprehensive Study on the Evaluation of Silhouette-based Gait Recognition](https://ieeexplore.ieee.org/document/9928336).
    '''
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-01'], 'BG': ['bg-01'], 'CL': ['cl-01']},
                      'OUMVLP': {'NM': ['00']}}

    gallery_seq_dict = {'CASIA-B': ['nm-02', 'bg-02', 'cl-02'],
                        'OUMVLP': ['01']}

    msg_mgr = get_msg_mgr()
    acc = {}
    mean_ap = {}
    view_list = sorted(np.unique(view))
    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros(len(view_list)) - 1.
        mean_ap[type_] = np.zeros(len(view_list)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset])
            gallery_y = label[gseq_mask]
            gallery_x = feature[gseq_mask, :]
            dist = cuda_dist(probe_x, gallery_x, metric)
            eval_results = compute_ACC_mAP(
                dist.cpu().numpy(), probe_y, gallery_y, view[pseq_mask], view[gseq_mask])
            acc[type_][v1] = np.round(eval_results[0] * 100, 2)
            mean_ap[type_][v1] = np.round(eval_results[1] * 100, 2)

    result_dict = {}
    msg_mgr.log_info(
        '===Cross View Gallery Evaluation (Excluded identical-view cases)===')
    out_acc_str = "========= Rank@1 Acc =========\n"
    out_map_str = "============= mAP ============\n"
    for type_ in probe_seq_dict[dataset].keys():
        avg_acc = np.mean(acc[type_])
        avg_map = np.mean(mean_ap[type_])
        result_dict[f'scalar/test_accuracy/{type_}-Rank@1'] = avg_acc
        result_dict[f'scalar/test_accuracy/{type_}-mAP'] = avg_map
        out_acc_str += f"{type_}:\t{acc[type_]}, mean: {avg_acc:.2f}%\n"
        out_map_str += f"{type_}:\t{mean_ap[type_]}, mean: {avg_map:.2f}%\n"
    # msg_mgr.log_info(f'========= Rank@1 Acc =========')
    msg_mgr.log_info(f'{out_acc_str}')
    # msg_mgr.log_info(f'========= mAP =========')
    msg_mgr.log_info(f'{out_map_str}')
    return result_dict

# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py


def single_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},
                      'OUMVLP': {'NM': ['00']},
                      'CASIA-E': {'NM': ['H-scene2-nm-1', 'H-scene2-nm-2', 'L-scene2-nm-1', 'L-scene2-nm-2', 'H-scene3-nm-1', 'H-scene3-nm-2', 'L-scene3-nm-1', 'L-scene3-nm-2', 'H-scene3_s-nm-1', 'H-scene3_s-nm-2', 'L-scene3_s-nm-1', 'L-scene3_s-nm-2', ],
                                  'BG': ['H-scene2-bg-1', 'H-scene2-bg-2', 'L-scene2-bg-1', 'L-scene2-bg-2', 'H-scene3-bg-1', 'H-scene3-bg-2', 'L-scene3-bg-1', 'L-scene3-bg-2', 'H-scene3_s-bg-1', 'H-scene3_s-bg-2', 'L-scene3_s-bg-1', 'L-scene3_s-bg-2'],
                                  'CL': ['H-scene2-cl-1', 'H-scene2-cl-2', 'L-scene2-cl-1', 'L-scene2-cl-2', 'H-scene3-cl-1', 'H-scene3-cl-2', 'L-scene3-cl-1', 'L-scene3-cl-2', 'H-scene3_s-cl-1', 'H-scene3_s-cl-2', 'L-scene3_s-cl-1', 'L-scene3_s-cl-2']
                                  },
                      'SUSTech1K': {'Normal': ['01-nm'], 'Bag': ['bg'], 'Clothing': ['cl'], 'Carrying':['cr'], 'Umberalla': ['ub'], 'Uniform': ['uf'], 'Occlusion': ['oc'],'Night': ['nt'], 'Overall': ['01','02','03','04']}
                      }
    gallery_seq_dict = {'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                        'OUMVLP': ['01'],
                        'CASIA-E': ['H-scene1-nm-1', 'H-scene1-nm-2', 'L-scene1-nm-1', 'L-scene1-nm-2'],
                        'SUSTech1K': ['00-nm'],}
    msg_mgr = get_msg_mgr()
    acc = {}
    view_list = sorted(np.unique(view))
    num_rank = 1
    if dataset == 'CASIA-E':
        view_list.remove("270")
    if dataset == 'SUSTech1K':
        num_rank = 5 
    view_num = len(view_list)

    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros((view_num, view_num, num_rank)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            pseq_mask = pseq_mask if 'SUSTech1K' not in dataset   else np.any(np.asarray(
                        [np.char.find(seq_type, probe)>=0 for probe in probe_seq]), axis=0
                            ) & np.isin(view, probe_view) # For SUSTech1K only
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset]) & np.isin(
                    view, [gallery_view])
                gseq_mask = gseq_mask if 'SUSTech1K' not in dataset  else np.any(np.asarray(
                            [np.char.find(seq_type, gallery)>=0 for gallery in gallery_seq_dict[dataset]]), axis=0
                                ) & np.isin(view, [gallery_view]) # For SUSTech1K only
                gallery_y = label[gseq_mask]
                gallery_x = feature[gseq_mask, :]
                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                acc[type_][v1, v2, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                                     0) * 100 / dist.shape[0], 2)

    result_dict = {}
    msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
    out_str = ""
    global_mean = 0.
    for rank in range(num_rank):
        out_str = ""
        for type_ in probe_seq_dict[dataset].keys():
            sub_acc = de_diag(acc[type_][:,:,rank], each_angle=True)
            if rank == 0:
                msg_mgr.log_info(f'{type_}@R{rank+1}: {sub_acc}')
                result_dict[f'scalar/test_accuracy/{type_}@R{rank+1}'] = np.mean(sub_acc)
                global_mean += result_dict[f'scalar/test_accuracy/{type_}@R{rank + 1}']
            out_str += f"{type_}@R{rank+1}: {np.mean(sub_acc):.2f}%\t"
        # Add global mean
        global_mean /= len(probe_seq_dict[dataset])
        out_str += f"Mean: {global_mean:.3f}%\t"
        msg_mgr.log_info(out_str)
    return result_dict


def evaluate_indoor_dataset(data, dataset, metric='euc', cross_view_gallery=False):
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)

    if dataset not in ('CASIA-B', 'OUMVLP', 'CASIA-E', 'SUSTech1K'):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    if cross_view_gallery:
        return cross_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)
    else:
        return single_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)

def evaluate_CCPG(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']

    label = np.array(label)
    for i in range(len(view)):
        view[i] = view[i].split("_")[0]
    view_np = np.array(view)
    view_list = list(set(view))
    view_list.sort()

    view_num = len(view_list)

    probe_seq_dict = {'CCPG': [["U0_D0_BG", "U0_D0"], [
        "U3_D3"], ["U1_D0"], ["U0_D0_BG"]]}

    gallery_seq_dict = {
        'CCPG': [["U1_D1", "U2_D2", "U3_D3"], ["U0_D3"], ["U1_D1"], ["U0_D0"]]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                   view_num, view_num, num_rank]) - 1.

    ap_save = []
    cmc_save = []
    minp = []
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        gseq_mask = np.isin(seq_type, gallery_seq)
        gallery_x = feature[gseq_mask, :]
        # print("gallery_x", gallery_x.shape)
        gallery_y = label[gseq_mask]
        gallery_view = view_np[gseq_mask]

        pseq_mask = np.isin(seq_type, probe_seq)
        probe_x = feature[pseq_mask, :]
        probe_y = label[pseq_mask]
        probe_view = view_np[pseq_mask]

        msg_mgr.log_info(
            ("gallery length", len(gallery_y), gallery_seq, "probe length", len(probe_y), probe_seq))
        distmat = cuda_dist(probe_x, gallery_x, metric).cpu().numpy()
        # cmc, ap = evaluate(distmat, probe_y, gallery_y, probe_view, gallery_view)
        cmc, ap, inp = evaluate_many(
            distmat, probe_y, gallery_y, probe_view, gallery_view)
        ap_save.append(ap)
        cmc_save.append(cmc[0])
        minp.append(inp)

    # print(ap_save, cmc_save)

    msg_mgr.log_info(
        '===Rank-1 (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
        cmc_save[0]*100, cmc_save[1]*100, cmc_save[2]*100, cmc_save[3]*100))

    msg_mgr.log_info(
        '===mAP (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
        ap_save[0]*100, ap_save[1]*100, ap_save[2]*100, ap_save[3]*100))

    msg_mgr.log_info(
        '===mINP (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                     (minp[0]*100, minp[1]*100, minp[2]*100, minp[3]*100))

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        for (v1, probe_view) in enumerate(view_list):
            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                    view, [gallery_view])
                gallery_x = feature[gseq_mask, :]
                gallery_y = label[gseq_mask]

                pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                    view, [probe_view])
                probe_x = feature[pseq_mask, :]
                probe_y = label[pseq_mask]

                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.sort(1)[1].cpu().numpy()
                # print(p, v1, v2, "\n")
                acc[p, v1, v2, :] = np.round(
                    np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                           0) * 100 / dist.shape[0], 2)
    result_dict = {}
    for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d (Include identical-view cases)===' % (i + 1))
        msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i]),
            np.mean(acc[3, :, :, i])))
    for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
        msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
            de_diag(acc[0, :, :, i]),
            de_diag(acc[1, :, :, i]),
            de_diag(acc[2, :, :, i]),
            de_diag(acc[3, :, :, i])))
    result_dict["scalar/test_accuracy/CL"] = acc[0, :, :, i]
    result_dict["scalar/test_accuracy/UP"] = acc[1, :, :, i]
    result_dict["scalar/test_accuracy/DN"] = acc[2, :, :, i]
    result_dict["scalar/test_accuracy/BG"] = acc[3, :, :, i]
    np.set_printoptions(precision=2, floatmode='fixed')
    for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
        msg_mgr.log_info('CL: {}'.format(de_diag(acc[0, :, :, i], True)))
        msg_mgr.log_info('UP: {}'.format(de_diag(acc[1, :, :, i], True)))
        msg_mgr.log_info('DN: {}'.format(de_diag(acc[2, :, :, i], True)))
        msg_mgr.log_info('BG: {}'.format(de_diag(acc[3, :, :, i], True)))
    return result_dict

def evaluation_fvg_b(data, dataset, metric='euc', num_rank=1, threshold=[0.01, 0.05, 0.1]):
    assert dataset == 'FVG-B', f'Running FVG-B evaluation for wrong dataset {dataset}'

    def find_idx(fpr, tpr, threshold=[0.01, 0.05, 0.1], ifround=True, eps=0.005):
        output = []
        for i in threshold:
            item = fpr[fpr < (i + eps)].max()
            idx = np.where(fpr == item)
            val = tpr[idx][-1]
            if ifround:
                val = round(val, 2)
            output.append(val)
        return output

    msg_mgr = get_msg_mgr()
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)

    # Retrieve session and #seq from seq_type
    split = [s.split('-')  for s in seq_type]
    sessions = np.array([row[0] for row in split])
    seq_num = np.array([row[1] for row in split])

    benchmarks = {
                    'WS': {
                            'session1': {
                                    'gallery_seqs': ['02'],
                                    'probe_seqs': ['04', '05', '06', '07', '08', '09']
                                },
                            'session2': {
                                    'gallery_seqs': ['02'],
                                    'probe_seqs': ['04', '05', '06']
                                },
                        },
                    'BGHT': {
                            'session1': {
                                    'gallery_seqs': ['02'],
                                    'probe_seqs': ['10', '11', '12']
                                }
                        },
                    'CL': {
                            'session2': {
                                    'gallery_seqs': ['02'],
                                    'probe_seqs': ['07', '08', '09']
                                }
                        },
                    'MP': {
                            'session2': {
                                    'gallery_seqs': ['02'],
                                    'probe_seqs': ['10', '11', '12']
                                }
                        },
                    'ALL': {
                            'session1': {
                                    'gallery_seqs': ['02'],
                                    'probe_seqs': ['01', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
                                },
                            'session2': {
                                    'gallery_seqs': ['02'],
                                    'probe_seqs': ['01', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
                                },
                            'session3': {
                                    'gallery_seqs': [],
                                    'probe_seqs': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
                                },
                        }
                }
    sub_per_bench = {'WS': 90, 'BGHT': 60, 'CL': 30, 'MP': 30, 'ALL': 90}
    valid_benchmarks = ("WS", "BGHT", "CL", "MP", "ALL")

    acc = {test: np.zeros(num_rank) - 1. for test in benchmarks}
    tar = {test: np.zeros(num_rank) - 1. for test in benchmarks}
    num_probe_seqs = {test: 0 for test in valid_benchmarks}
    num_gallery_seqs = {test: 0 for test in valid_benchmarks}


    for i, test in enumerate(benchmarks):
        # Reset selection mask
        gseq_mask = np.zeros(len(feature), dtype=bool)
        pseq_mask = np.zeros(len(feature), dtype=bool)

        # Compute selection mask
        for sess in benchmarks[test]:
            gallery_seqs = benchmarks[test][sess]['gallery_seqs'] 
            probe_seqs = benchmarks[test][sess]['probe_seqs']

            gseq_mask |= ((sessions == sess) & np.isin(seq_num, gallery_seqs))
            pseq_mask |= ((sessions == sess) & np.isin(seq_num, probe_seqs))

        gallery_x = feature[gseq_mask, :]
        gallery_y = label[gseq_mask]

        probe_x = feature[pseq_mask, :]
        probe_y = label[pseq_mask]

        # Count number of seqs
        if test in valid_benchmarks:
            num_probe_seqs[test] = pseq_mask.sum()
            num_gallery_seqs[test] = gseq_mask.sum()

        dist = cuda_dist(probe_x, gallery_x, metric)

        idx = dist.sort(1)[1].cpu().numpy()
        gallery_seq_type = np.array(seq_type)[gseq_mask]
        probe_seq_type = np.array(seq_type)[pseq_mask]
        comp = zip(gallery_y[idx[:, 0]],
                    gallery_seq_type[idx[:, 0]],
                    probe_y,
                    probe_seq_type,
                    dist.max(dim=1)[0].cpu().numpy())

        # Accuracy computation
        acc[test] = np.round(
            np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                   0) * 100 / dist.shape[0], 2)

        # TAR@FAR computation
        # Normalize dist over gallery dim so to stabilize computation
        if metric == 'euc':
            sim = 1. - (dist / dist.max(dim=1, keepdims=True)[0]).cpu().numpy()
        else:
            sim = (1. - dist).cpu().numpy()

        # Consider performance for a binnary problem, considering:
        # class 1: Correct pair matching - class 0: incorrect pair matching
        gt = make_bin_labels(probe_y, gallery_y) # probe label vs gallery label

        assert sim.shape == gt.shape, f'Not matching sim and gt labels'

        sim = sim.flatten()
        gt = gt.flatten()

        fpr, tpr, _ = roc_curve(gt, sim)
        # TODO in future ver.: Include ROC curve plotting
        tar[test] = find_idx(fpr, tpr, threshold, ifround=False) # tar[i, :]

    # Account total number of seqs
    num_comps = {test: (num_probe_seqs[test] * num_gallery_seqs[test]) for test in valid_benchmarks}
    total_num_comps = sum(num_comps.values())
    sum_num_sub_per_bench = sum(sub_per_bench.values())

    result_dict = {}
    np.set_printoptions(precision=3, suppress=True)
    for test_idx, test in enumerate(benchmarks):
        if test in valid_benchmarks:
            msg_mgr.log_info(
                    #'############################################'\
                    f'############# Benchmark {test} ############# Probe seqs: {num_probe_seqs[test]}, Gallery seqs: {num_gallery_seqs[test]}, # subjects {sub_per_bench[test]}  ####')
                    #'############################################')
        else:
            msg_mgr.log_info(
                    f'############# Benchmark {test} ################################################')
        for i in range(num_rank):
            msg_mgr.log_info(
                f'===Top-{i + 1} accuracy (Include identical-view cases): {acc[test][i]:.3f} ==='
            )

        for i, thr in enumerate(threshold):
            msg_mgr.log_info(
                f'===TAR@FAR {thr * 100} % (Include identical-view cases): {tar[test][i] * 100:.3f} % ==='
            )

        result_dict[f"scalar/test_accuracy/{test}"] = acc[test][0]
 
    msg_mgr.log_info(
        f'----- Weighted global mean (per # of subj.) acc and TAR@FAR ----')
    # Compute global (weighted) acc and tar mean
    for i in range(num_rank):
        msg_mgr.log_info(f'### Top-{i + 1} mean norm accuracy: {sum(sub_per_bench[test] * acc[test][i] / sum_num_sub_per_bench for test in valid_benchmarks ):.3f}')
    
    for i, thr in enumerate(threshold):
        msg_mgr.log_info(f'### TAR@FAR {thr * 100} % mean norm accuracy: {sum(sub_per_bench[test] * tar[test][i] / sum_num_sub_per_bench for test in valid_benchmarks ) * 100:.3f}')   
    
    np.set_printoptions(precision=2, floatmode='fixed')

    return result_dict
