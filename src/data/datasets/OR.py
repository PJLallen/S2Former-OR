# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
dataset (COCO-like) which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import json
import torch
import torch.utils.data
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask
import random
import src.data.transforms.transforms as T
from torch.nn.functional import one_hot
import os.path
from typing import Any, Callable, List, Optional, Tuple
import open3d as o3d
from PIL import Image
import numpy as np
from torchvision.datasets.vision import VisionDataset

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0] < num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

class CocoDetection_infer(Dataset):
    def __init__(self, infer_folder, transforms):
        super().__init__()
        self._transforms = transforms
        self.img_folder = os.path.join(infer_folder, "images_infer")
        self.pcd_folder = os.path.join(infer_folder, "points_infer")
        total_list = os.listdir(self.img_folder)
        total_list.sort()
        self.multiview_list = []
        self.views = [2, 3, 6]
        self.img_list = [inst for inst in total_list if "cam_1" in inst]
        for k in self.img_list:
            self.multiview_list.append([inst for inst in total_list if (k.split("_cam")[0] in inst and "cam_1" not in inst and int(inst.split("_")[-1].split(".")[0]) in self.views)])
        self.img_list.sort()
        self.multiview_list.sort()
        # print("total list", total_list)
        # print("img list", self.img_list)
        # print("view list", self.multiview_list)

    def __getitem__(self, idx):
        image_id = self.img_list[idx]
        multiview_ids = self.multiview_list[idx]
        img = Image.open(os.path.join(self.img_folder, image_id)).convert("RGB")
        images_multiview = [Image.open(os.path.join(self.img_folder, id)).convert("RGB") for id in multiview_ids]
        points_path = os.path.join(self.pcd_folder, image_id.split("_cam")[0]+".pcd")
        pcd = o3d.io.read_point_cloud(points_path)
        point_cloud = np.concatenate([np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1)
        # scaling
        point_cloud[:, :3] /= 1000
        point_cloud[:, 3:] = (point_cloud[:, 3:] - np.array([0.49, 0.54, 0.58]))
        point_cloud, choices = random_sampling(point_cloud, 200000, return_choices=True)
        point_cloud = torch.tensor(point_cloud).type(torch.FloatTensor)

        if self._transforms is not None:
            img, _, images_multiview = self._transforms(img, None, images_multiview)

        return img, image_id, images_multiview, point_cloud

    def __len__(self):
        return len(self.img_list)




class MultiView_CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.views = [2, 3, 6]

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_points(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        points_path = os.path.join(self.root.replace("images", "points"), path.replace("jpg", "pcd"))
        pcd = o3d.io.read_point_cloud(points_path)
        point_cloud = np.concatenate([np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1)
        # scaling
        point_cloud[:, :3] /= 1000
        point_cloud[:, 3:] = (point_cloud[:, 3:] - np.array([0.49, 0.54, 0.58]))
        point_cloud, choices = random_sampling(point_cloud, 200000, return_choices=True)
        point_cloud = torch.tensor(point_cloud).type(torch.FloatTensor)
        return point_cloud

    def _load_image_multiview(self, id: int):
        path_mainview = self.coco.loadImgs(id)[0]["file_name"]
        return [Image.open(os.path.join(self.root, path_mainview.split('.')[0] + '_view' + str(view) + '.' + path_mainview.split('.')[1])).convert("RGB") for view in self.views]

    def _load_image_video(self, id: int):
        if id-1 < min(self.ids) or id+1 > max(self.ids):
            return None
        else:
            video_ids = [id-1, id+1]
            path_video = [self.coco.loadImgs(id)[0]["file_name"] for id in video_ids]
            return [Image.open(os.path.join(self.root, path)).convert("RGB") for path in path_video]


    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        images_multiview = self._load_image_multiview(id)
        images_video = self._load_image_video(id)
        points = self._load_points(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            images_multiview = [self.transforms(i) for i in images_multiview]
            images_video = [self.transforms(j) for j in images_video]

        return image, target, images_multiview, points, images_video

    def __len__(self) -> int:
        return len(self.ids)

class CocoDetection(MultiView_CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, args):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.num_actions = args.num_actions

        #TODO load relationship
        with open('/'.join(ann_file.split('/')[:-1])+'/rel.json', 'r') as f:
            all_rels = json.load(f)
        if 'train' in ann_file:
            self.rel_annotations = all_rels['train']
        elif 'val' in ann_file:
            self.rel_annotations = all_rels['val']
        else:
            self.rel_annotations = all_rels['val']

        self.rel_categories = all_rels['rel_categories']

    def __getitem__(self, idx):
        img, target, images_multiview, points, images_video = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        while not self.rel_annotations[str(image_id)]:
            idx = random.randint(0, len(self.ids)-1)
            image_id = self.ids[idx]
            img, target, images_multiview, points, images_video = super(CocoDetection, self).__getitem__(idx)

        rel_target = self.rel_annotations[str(image_id)]

        target = {'image_id': image_id, 'annotations': target, 'rel_annotations': rel_target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target, images_multiview, images_video = self._transforms(img, target, images_multiview, images_video)
        target["hoi_labels"] = one_hot(torch.cat([target["rel_annotations"][:, 2]]), num_classes=self.num_actions).type(torch.float32)
        target["obj_labels"] = torch.cat([target['labels'][target['rel_annotations'][:, 1]]])
        target["sub_labels"] = torch.cat([target['labels'][target['rel_annotations'][:, 0]]])
        sub_boxes = torch.cat([target['boxes'][target['rel_annotations'][:, 0]]])
        obj_boxes = torch.cat([target['boxes'][target['rel_annotations'][:, 1]]])
        target['sub_boxes'] = sub_boxes
        target['obj_boxes'] = obj_boxes
        target['gt_triplet'] = torch.cat([target["sub_labels"].unsqueeze(-1), target["obj_labels"].unsqueeze(-1), torch.cat([target["rel_annotations"][:, 2]]).unsqueeze(-1)], dim=1)


        # relation map
        kept_box_indices = []
        for _ in range(target['boxes'].shape[0]):
            kept_box_indices.append(_)
        sub_obj_pairs = [(inst[0], inst[1])for inst in target["rel_annotations"]]
        relation_map = torch.zeros((len(target['boxes']), len(target['boxes']), self.num_actions))
        for sub_obj_pair in sub_obj_pairs:
            kept_subj_id = kept_box_indices.index(sub_obj_pair[0])
            kept_obj_id = kept_box_indices.index(sub_obj_pair[1])
            relation_map[kept_subj_id, kept_obj_id] = target["hoi_labels"][sub_obj_pairs.index(sub_obj_pair), :].clone().detach()
        target['relation_map'] = relation_map
        target['hois'] = relation_map.nonzero(as_tuple=False)

        if not images_video:
            images_video = [torch.zeros_like(img), torch.zeros_like(img)]

        return img, target, images_multiview, points, images_video


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        keep = (boxes[:, 3] >= boxes[:, 1]) & (boxes[:, 2] >= boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # TODO add relation gt in the target
        rel_annotations = target['rel_annotations']

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # TODO add relation gt in the target
        target['rel_annotations'] = torch.tensor(rel_annotations)

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    # T.RandomSizeCrop(384, 600), # TODO: cropping causes that some boxes are dropped then no tensor in the relation part! What should we do?
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize])

    if image_set == 'val' or "test" or "infer":
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):

    ann_path = args.ann_path
    img_folder = args.img_folder

    if image_set == 'train':
        ann_file = ann_path + 'train.json'
    elif image_set == 'val':
        ann_file = ann_path + 'val.json'
    elif image_set == 'infer':
        img_folder = args.img_folder_infer
        dataset = CocoDetection_infer(img_folder, transforms=make_coco_transforms(image_set))
        return dataset
    # elif image_set == 'val' or image_set == 'test':
    #     if args.eval:
    #         ann_file = ann_path + 'test.json'
    #     else:
    #         ann_file = ann_path + 'val.json'

    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=False, args=args)
    return dataset
