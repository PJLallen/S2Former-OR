import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report
import src.util.misc as utils
import src.util.logger as loggers
from src.data.evaluators.or_eval import OREvaluator
from src.models.or_utils import check_annotation, plot_cross_attention, plot_hoi_results
import json

OBJECT_LABEL_MAP = {
    0: 'anesthesia_equipment',
    1: 'operating_table',
    2: 'instrument_table',
    3: 'secondary_table',
    4: 'instrument',
    5: 'Patient',
    6: 'human_0',
    7: 'human_1',
    8: 'human_2',
    9: 'human_3',
    10: 'human_4',
}
VERB_LABEL_MAP = {
    0: "Assisting",
    1: "Cementing",
    2: "Cleaning",
    3: "CloseTo",
    4: "Cutting",
    5: "Drilling",
    6: "Hammering",
    7: "Holding",
    8: "LyingOn",
    9: "Operating",
    10: "Preparing",
    11: "Sawing",
    12: "Suturing",
    13: "Touching",
}
VERB_LABEL_MAP_None = {
    0: "Assisting",
    1: "Cementing",
    2: "Cleaning",
    3: "CloseTo",
    4: "Cutting",
    5: "Drilling",
    6: "Hammering",
    7: "Holding",
    8: "LyingOn",
    9: "Operating",
    10: "Preparing",
    11: "Sawing",
    12: "Suturing",
    13: "Touching",
    14: "None"
}


@torch.no_grad()
def or_evaluate(model, postprocessors, data_loader, device, thr, args):
    model.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (HICO-DET)'

    preds = []
    gts = []
    indices = []
    hoi_recognition_time = []
    names = []

    for samples, targets, multiview_samples, points, _ in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        multiview_samples = multiview_samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        points = torch.cat([p.unsqueeze(0) for p in points], dim=0).to(device)

        outputs = model(samples, None, multiview_samples, points)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes, threshold=thr, dataset='or')
        hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

        # # visualize
        # if targets[0]['id'] in [57]: # [47, 57, 81, 30, 46, 97]: # 30, 46, 97
        #     # check_annotation(samples, targets, mode='eval', rel_num=20)
        #
        #     # visualize cross-attentioa
        #     if 'HOTR' in type(model).__name__:
        #         outputs['pred_actions'] = outputs['pred_actions'][:, :, :args.num_actions]
        #         outputs['pred_rel_pairs'] = [x.cpu() for x in torch.stack([outputs['pred_hidx'].argmax(-1), outputs['pred_oidx'].argmax(-1)], dim=-1)]
        #     topk_qids, q_name_list = plot_hoi_results(samples, outputs, targets, args=args)
        #     plot_cross_attention(samples, outputs, targets, dec_crossattn_weights, topk_qids=topk_qids)
        #     print(f"image_id={targets[0]['id']}")
        #
        #     # visualize self attention
        #     print('visualize self-attention')
        #     q_num = len(dec_selfattn_weights[0][0])
        #     plt.figure(figsize=(10,4))
        #     plt.imshow(dec_selfattn_weights[0][0].cpu().numpy(), vmin=0, vmax=0.4)
        #     plt.xticks(np.arange(q_num), [f"{i}" for i in range(q_num)], rotation=90, fontsize=12)
        #     plt.yticks(np.arange(q_num), [f"({q_name_list[i]})={i}" for i in range(q_num)], fontsize=12)
        #     plt.gca().xaxis.set_ticks_position('top')
        #     plt.grid(alpha=0.4, linestyle=':')
        #     plt.show()
        # hook_self.remove(); hook_cross.remove()

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # if len(gts) >= 2:
        #     break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [int(img_gts['image_id']) for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    # now 4DOR evaluation!
    OR_GT = []
    OR_PRED = []
    eval_dict = {}
    eval_dict_gt = {}
    for iter in range(len(gts)):
        or_gt_img = []
        or_pred_img = []
        gt_pair_collection = []
        gt_labels_sop = gts[iter]['gt_triplet']
        det_labels_sop_top = preds[iter]['triplet']
        det_scores = preds[iter]['ranked_scores']
        name = gts[iter]['image_id']
        scores_matched = []

        if args.add_none:
            all_pairs = torch.cat(
                [torch.cat([gts[iter]['labels'].unsqueeze(-1), gts[iter]['labels'].roll(i + 1).unsqueeze(-1)], dim=1)
                 for i in range(len(gts[iter]['labels']) - 1)], dim=0)
            all_pairs = torch.cat([all_pairs, (torch.zeros(all_pairs.shape[0], 1) + 14).to(all_pairs.device)], dim=1)
            for k in range(all_pairs.shape[0]):
                pair = all_pairs[k]
                for m in range(gt_labels_sop.shape[0]):
                    tmp = gt_labels_sop[m]
                    if tmp[0] == pair[0] and tmp[1] == pair[1]:
                        all_pairs[k] = tmp
            gt_labels_sop = all_pairs

        for index in range(gt_labels_sop.shape[0]):
            found = False
            if (gt_labels_sop[index][0], gt_labels_sop[index][1]) not in gt_pair_collection:
                gt_pair_collection.append((gt_labels_sop[index][0], gt_labels_sop[index][1]))
                or_gt_img.append(gt_labels_sop[index][2])
                for idx in range(len(det_labels_sop_top)):
                    if args.use_tricks:
                        if det_labels_sop_top[idx][2] == 8 and (
                                det_labels_sop_top[idx][0] != 5 or det_labels_sop_top[idx][1] != 1):
                            continue
                        if det_labels_sop_top[idx][2] == 9 and (
                                (det_labels_sop_top[idx][0] not in [6, 7]) or det_labels_sop_top[idx][1] != 1):
                            continue
                        if ((det_labels_sop_top[idx][0] not in [6, 7]) or (det_labels_sop_top[idx][1] != 5)) and (
                                det_labels_sop_top[idx][2] in [1, 2, 4, 5, 6, 11, 12]):
                            continue
                        if ((det_labels_sop_top[idx][0] not in [6, 7, 8]) or (det_labels_sop_top[idx][1] != 4)) and (
                                det_labels_sop_top[idx][2] == 7):
                            continue
                        if ((det_labels_sop_top[idx][0] != 7) or (det_labels_sop_top[idx][1] != 6)) and (
                                det_labels_sop_top[idx][2] == 0):
                            continue
                        if (not ((det_labels_sop_top[idx][0] == 6 and det_labels_sop_top[idx][1] == 5) or (
                                det_labels_sop_top[idx][0] == 7 and det_labels_sop_top[idx][1] == 5))) and (
                                det_labels_sop_top[idx][2] == 10):
                            continue
                        if (not ((det_labels_sop_top[idx][0] == 7 and det_labels_sop_top[idx][1] == 2) or (
                                det_labels_sop_top[idx][0] == 8 and det_labels_sop_top[idx][1] == 3))) and (
                                det_labels_sop_top[idx][2] == 13):
                            continue
                    elif args.use_tricks_val:
                        if det_labels_sop_top[idx][0] == det_labels_sop_top[idx][1]:
                            continue
                        if det_labels_sop_top[idx][2] in [1, 4, 5, 6, 11, 12] and (
                                det_labels_sop_top[idx][0] != 6 or det_labels_sop_top[idx][1] != 5):
                            continue
                        if det_labels_sop_top[idx][2] == 8 and (
                                det_labels_sop_top[idx][0] != 5 or det_labels_sop_top[idx][1] != 1):
                            continue
                        if det_labels_sop_top[idx][2] == 9 and (
                                (det_labels_sop_top[idx][0] not in [6, 7]) or det_labels_sop_top[idx][1] != 1):
                            continue
                        if ((det_labels_sop_top[idx][0] not in [6, 7]) or (det_labels_sop_top[idx][1] != 5)) and (
                                det_labels_sop_top[idx][2] in [2, 10]):
                            continue
                        if ((det_labels_sop_top[idx][0] not in [6, 7]) or (det_labels_sop_top[idx][1] != 4)) and (
                                det_labels_sop_top[idx][2] == 7):
                            continue
                        # if ((det_labels_sop_top[idx][0] != 7) or (det_labels_sop_top[idx][1] != 6)) and (
                        #         det_labels_sop_top[idx][2] == 0):
                        #     continue
                        if ((det_labels_sop_top[idx][0] not in [6, 7]) or (
                                det_labels_sop_top[idx][1] not in [6, 7])) and (
                                det_labels_sop_top[idx][2] == 0):
                            continue
                        if (not ((det_labels_sop_top[idx][0] == 7 and det_labels_sop_top[idx][1] == 2) or (
                                det_labels_sop_top[idx][0] == 8 and det_labels_sop_top[idx][1] == 3))) and (
                                det_labels_sop_top[idx][2] == 13):
                            continue
                        if (det_labels_sop_top[idx][0] not in [5, 6, 7, 8, 9]) and (
                                det_labels_sop_top[idx][2] != 3):
                            continue

                    if gt_labels_sop[index][0] == det_labels_sop_top[idx][0] and gt_labels_sop[index][1] == \
                            det_labels_sop_top[idx][1]:
                        or_pred_img.append(det_labels_sop_top[idx][2])
                        scores_matched.append(det_scores[idx])
                        found = True
                        break
                if not found:
                    or_pred_img.append(torch.tensor(14))
                    scores_matched.append(torch.tensor(0))
        OR_GT.extend(or_gt_img)
        #       Assisting filter
        if args.use_tricks_val:
            for m, k in enumerate(or_pred_img):
                if k == 3 and scores_matched[m] < 0.08:
                    or_pred_img[m] = torch.tensor(14)
                if k == 0:
                    if scores_matched[m] < 0.08:
                        or_pred_img[m] = torch.tensor(14)
                    else:
                        hold = False
                        rest = True
                        sub = gt_labels_sop[m][1]
                        for o, p in enumerate(or_pred_img):
                            if p == 7 and gt_labels_sop[o][0] == sub:
                                hold = True
                                break
                        # for o, p in enumerate(or_pred_img):
                        #     if p not in [3, 8]:
                        #         rest = False
                        #         break
                        if not hold:
                            or_pred_img[m] = torch.tensor(14)
        eval_dict[int(name)] = [VERB_LABEL_MAP_None[int(j)] for j in or_pred_img]
        eval_dict_gt[int(name)] = [VERB_LABEL_MAP_None[int(m)] for m in or_gt_img]
        OR_PRED.extend(or_pred_img)
        OR_GT = [inst.cpu() for inst in OR_GT]

    with open("eval_dict.json", 'w') as f:
        json.dump(eval_dict, f)
    with open("eval_dict_gt.json", 'w') as f:
        json.dump(eval_dict_gt, f)
    cls_report = classification_report(OR_GT, OR_PRED,
                                       target_names=["Assisting", "Cementing", "Cleaning", "CloseTo", "Cutting",
                                                     "Drilling", "Hammering", "Holding", "LyingOn", "Operating",
                                                     "Preparing", "Sawing", "Suturing", "Touching", "None"],
                                       output_dict=True)
    print(cls_report)

    # if args.infer_val:
    #     pass

    return


def or_evaluate_infer(model, postprocessors, data_loader, device, thr, args):
    model.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (HICO-DET)'

    preds = []
    names = []

    for samples, name, multiview_samples, points in metric_logger.log_every(data_loader, 50, header):
        # for samples, name, multiview_samples, points in data_loader:
        samples = samples.to(device)
        multiview_samples = multiview_samples.to(device)
        points = torch.cat([p.unsqueeze(0) for p in points], dim=0).to(device)

        outputs = model(samples, multiview_samples=multiview_samples, points=points)
        results = postprocessors['hoi'](outputs, None, threshold=thr, dataset='or')
        # outputs = model(samples, None, multiview_samples, points)
        # # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['hoi'](outputs, None, threshold=thr, dataset='or')

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        names.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(name)))))
        # preds.extend(results)
        # names.extend(name)

        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(name)
        # # print("samples:", samples)
        # print("multiview_samples:", multiview_samples.tensors[0])
        # # print("points:", points)
        # # print("outputs:", outputs)
        # print("scores::", results[0]['ranked_scores'])
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # if len(names) >= 20:
        #     break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    final_dict = {}
    final_dict2 = {}
    for idx in range(len(names)):
        relations = []
        name = names[idx].split("cam_")[0] + "2"
        name2 = names[idx].split("cam_")[0] + "1"
        sub_obj_pair_save = []
        scores = preds[idx]['ranked_scores']
        scores_matched = []
        for index in range(preds[idx]["triplet"].shape[0]):
            inst = preds[idx]["triplet"][index]
            sub = OBJECT_LABEL_MAP[int(inst[0])]
            obj = OBJECT_LABEL_MAP[int(inst[1])]
            verb = VERB_LABEL_MAP[int(inst[2])]
            if args.use_tricks_val:
                if inst[2] == 3 and scores[index] < 0.12:
                    continue
                if inst[2] == 7 and scores[index] < 0.05:
                    continue
                if inst[2] == 0:
                    if scores[index] < 0.15:
                        continue
                    else:
                        hold = False
                        rest = True
                        sub2 = inst[1]
                        for o in range(preds[idx]["triplet"].shape[0]):
                            inst2 = preds[idx]["triplet"][o]
                            if inst2[2] == 7 and inst2[0] == sub2:
                                hold = True
                                break
                        for p in range(preds[idx]["triplet"].shape[0]):
                            inst2 = preds[idx]["triplet"][p]
                            if inst2[2] not in [3, 8]:
                                rest = False
                                break
                        if (not hold) and (not rest):
                            continue

                if inst[0] == inst[1]:
                    continue
                if inst[2] in [1, 4, 5, 6, 11, 12] and (
                        inst[0] != 6 or inst[1] != 5):
                    continue
                if inst[2] == 8 and (
                        inst[0] != 5 or inst[1] != 1):
                    continue
                if inst[2] == 9 and (
                        (inst[0] not in [6, 7]) or inst[1] != 1):
                    continue
                if ((inst[0] not in [6, 7]) or (inst[1] != 5)) and (
                        inst[2] in [2, 10]):
                    continue
                if ((inst[0] not in [6, 7]) or (inst[1] != 4)) and (
                        inst[2] == 7):
                    continue
                if ((inst[0] not in [6, 7]) or (inst[1] not in [6, 7])) and (
                        inst[2] == 0):
                    continue
                if (not ((inst[0] == 7 and inst[1] == 2) or (
                        inst[0] == 8 and inst[1] == 3) or (inst[0] == 6 and inst[1] == 5) or (
                                 inst[0] == 7 and inst[1] == 5))) and (
                        inst[2] == 13):
                    continue
                if (inst[0] not in [5, 6, 7, 8, 9]) and (
                        inst[2] != 3):
                    continue

            if [sub, obj] not in sub_obj_pair_save and sub != obj:
                relations.append([sub, verb, obj])
                sub_obj_pair_save.append([sub, obj])
                scores_matched.append(scores[index])
            else:
                pass

        final_dict[name] = relations
        final_dict2[name2] = relations
    output_name = args.infer_name
    output_name2 = "backup.json"
    with open(output_name, 'w') as f:
        json.dump(final_dict, f)
    with open(output_name2, 'w') as f:
        json.dump(final_dict2, f)

    return