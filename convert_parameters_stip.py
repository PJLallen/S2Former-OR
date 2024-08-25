import argparse

import torch
from torch import nn


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load_path', type=str, required=True,
    )
    parser.add_argument(
        '--save_path', type=str, required=True,
    )
    parser.add_argument(
        '--dataset', type=str, default='hico',
    )
    parser.add_argument(
        '--num_queries', type=int, default=100,
    )

    args = parser.parse_args()

    return args


def main(args):
    ps = torch.load(r"params\r50_deformable_detr_single_scale-checkpoint.pth")

    # obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    #            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    #            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    #            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    #            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    #            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    #            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    #            82, 84, 85, 86, 87, 88, 89, 90]
    #
    # # For no pair
    # obj_ids.append(91)

    obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # For no pair
    obj_ids.append(11)

    for k in list(ps['model'].keys()):
        if "bbox_embed" in k or "class_embed" in k or "query_embed" in k:
            del ps['model'][k]
            print("delete embed")
        # if "backbone" in k:
        #     del ps['model'][k]

    # ps['model']['hum_bbox_embed.layers.0.weight'] = ps['model']['bbox_embed.layers.0.weight'].clone()
    # ps['model']['hum_bbox_embed.layers.0.bias'] = ps['model']['bbox_embed.layers.0.bias'].clone()
    # ps['model']['hum_bbox_embed.layers.1.weight'] = ps['model']['bbox_embed.layers.1.weight'].clone()
    # ps['model']['hum_bbox_embed.layers.1.bias'] = ps['model']['bbox_embed.layers.1.bias'].clone()
    # ps['model']['hum_bbox_embed.layers.2.weight'] = ps['model']['bbox_embed.layers.2.weight'].clone()
    # ps['model']['hum_bbox_embed.layers.2.bias'] = ps['model']['bbox_embed.layers.2.bias'].clone()
    #
    # ps['model']['obj_bbox_embed.layers.0.weight'] = ps['model']['bbox_embed.layers.0.weight'].clone()
    # ps['model']['obj_bbox_embed.layers.0.bias'] = ps['model']['bbox_embed.layers.0.bias'].clone()
    # ps['model']['obj_bbox_embed.layers.1.weight'] = ps['model']['bbox_embed.layers.1.weight'].clone()
    # ps['model']['obj_bbox_embed.layers.1.bias'] = ps['model']['bbox_embed.layers.1.bias'].clone()
    # ps['model']['obj_bbox_embed.layers.2.weight'] = ps['model']['bbox_embed.layers.2.weight'].clone()
    # ps['model']['obj_bbox_embed.layers.2.bias'] = ps['model']['bbox_embed.layers.2.bias'].clone()

    # ps['model']['obj_class_embed.weight'] = ps['model']['class_embed.weight'].clone()[obj_ids]
    # ps['model']['obj_class_embed.bias'] = ps['model']['class_embed.bias'].clone()[obj_ids]

    # ps['model']['query_embed.weight'] = ps['model']['query_embed.weight'].clone()[:args.num_queries]

    torch.save(ps, r"D:\DD\STIP_or\params\deformable-r50-pre-4dor-stip-noquery.pth")


if __name__ == '__main__':
    args = get_args()
    main(args)
