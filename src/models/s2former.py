import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util.misc import NestedTensor, nested_tensor_from_tensor_list
from torchvision.ops import roi_align
from .transformer import TransformerDecoderLayer, TransformerDecoder, TransformerDecoderLayer_multiview
from src.util import box_ops
import numpy as np
from src.util.misc import accuracy, is_dist_avail_and_initialized, get_world_size
import time
from .backbone_module import Pointnet2Backbone

class S2FORMER(nn.Module):
    def __init__(self, args, detr, detr_matcher):
        super().__init__()
        self.args = args
        self.detr_matcher = detr_matcher
        # * Instance Transformer ---------------
        self.detr = detr
        self.backbone_net = Pointnet2Backbone(input_feature_dim=3, width=1)
        if not args.train_detr:
            # if this flag is given, freeze the object detection related parameters of DETR
            for p in self.parameters():
                p.requires_grad_(False)
        # --------------------------------------

        # relation feature map
        if self.args.relation_feature_map_from == 'backbone':
            if args.use_high_resolution_relation_feature_map:
                relation_feature_map_dim = 1024
            else:
                relation_feature_map_dim = 2048
        elif self.args.relation_feature_map_from == 'detr_encoder':
            relation_feature_map_dim = self.args.hidden_dim

        # relation proposal
        rel_rep_dim = 1024
        self.coarse_relation_feature_extractor = RelationFeatureExtractor(args, in_channels=relation_feature_map_dim, out_dim=rel_rep_dim)
        # self.union_box_feature_extractor = RelationFeatureExtractor(args, in_channels=relation_feature_map_dim, out_dim=rel_rep_dim)
        self.relation_proposal_mlp = nn.Sequential(
            make_fc(rel_rep_dim, rel_rep_dim // 2), nn.ReLU(),
            make_fc(rel_rep_dim // 2, 1)
        )

        if self.args.backbone == "resnet34":
            self.backbone_channel_adapt = make_fc(512, 2048)

        # relation classification
        if self.args.use_simple_pointsfusion and self.args.use_pointsfusion:
            self.pointsfeats_proj = make_fc(291, rel_rep_dim)
        self.rel_query_pre_proj = make_fc(rel_rep_dim, self.args.hidden_dim)
        if self.args.no_interaction_decoder:
            self.args.hoi_aux_loss = False
        else:
            self.memory_input_proj = nn.Conv2d(relation_feature_map_dim, self.args.hidden_dim, kernel_size=1)
            if args.use_memory_layout_encoding:
                self.layout_embeddings = nn.Embedding(6, self.args.hidden_dim) # 0-pad, 1-image, 2-union, 3-subj, 4-obj, 5-intersection
                self.layout_content_aware_mapping = nn.Sequential(
                    make_fc(self.args.hidden_dim * 2, self.args.hidden_dim), nn.ReLU(),
                    make_fc(self.args.hidden_dim, self.args.hidden_dim)
                )
            if self.args.use_prior:
                self.prior_embeddings = nn.Embedding(7, self.args.hidden_dim)
            if self.args.use_relation_dependency_encoding:
                self.relation_dependency_embeddings = nn.Embedding(6, self.args.hidden_dim)
                self.relation_dependency_content_aware_mapping = nn.Sequential(
                    make_fc(self.args.hidden_dim * 2, self.args.hidden_dim), nn.ReLU(),
                    make_fc(self.args.hidden_dim, self.args.hidden_dim)
                )
            if self.args.use_query_fourier_encoding:
                self.fourier_feature_embedding = make_fc(1, self.args.hidden_dim//2) # group=8, group_dim=1
                self.fourier_mlp = nn.Sequential(
                    make_fc(self.args.hidden_dim, self.args.hidden_dim), nn.ReLU(),
                    make_fc(self.args.hidden_dim, self.args.hidden_dim // 8)
                )

            decoder_layer = TransformerDecoderLayer(d_model=self.args.hidden_dim, nhead=self.args.hoi_nheads)
            decoder_norm = nn.LayerNorm(self.args.hidden_dim)
            self.interaction_decoder = TransformerDecoder(decoder_layer, self.args.hoi_dec_layers, decoder_norm, return_intermediate=True)
        self.action_embed = nn.Linear(self.args.hidden_dim, self.args.num_actions)

    def forward(self, samples: NestedTensor, targets=None, multiview_samples=None, points=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # ################### visualiazation ###################
        # img_tensor = samples.tensors[0].permute(1, 2, 0).detach().cpu().numpy()
        # img_tensor = (img_tensor/2. + 1) / 2.
        # plt.imshow(img_tensor)
        # plt.savefig(f"ori.png")
        #
        # ################### visualiazation ###################

        start_time = time.time()
        # >>>>>>>>>>>>  BACKBONE LAYERS  <<<<<<<<<<<<<<<
        features, pos = self.detr.backbone(samples)
        features_multiview, pos_multiview = self.detr.backbone(multiview_samples)
        bs = features[-1].tensors.shape[0]

        if self.args.num_feature_levels > 1:
            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.detr.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
            if self.args.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.args.num_feature_levels):
                    if l == _len_srcs:
                        src = self.detr.input_proj[l](features[-1].tensors)
                    else:
                        src = self.detr.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)
        else:
            src, mask = features[-1].decompose()
            src_multiview, mask_multiview = features_multiview[-1].decompose()
            assert mask is not None
        # >>>>>>>>>>>>   POINT CLOUD  <<<<<<<<<<<<<<<
        end_points = {}
        if self.args.use_pointsfusion:
            end_points = self.backbone_net(points, end_points)
            point_features = torch.cat([end_points['fp2_features'].permute(0, 2, 1), end_points['fp2_xyz']], dim=-1)
        else:
            point_features = None

        if self.args.backbone == "resnet34":
            src = self.backbone_channel_adapt(src.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            src_multiview = self.backbone_channel_adapt(
                src_multiview.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        # >>>>>>>>>>>> OBJECT DETECTION LAYERS <<<<<<<<<<
        if self.args.deformable_detr:
            if self.args.num_feature_levels > 1:
                hs, _, detr_ref_outs, __, ___ = self.detr.transformer(srcs, masks, pos, self.detr.query_embed.weight,
                                                                      points_fusion=self.args.use_pointsfusion,
                                                                      point_features=point_features)
            else:
                hs, _, detr_ref_outs, __, ___ = self.detr.transformer(self.detr.input_proj[0](src), mask, pos[-1], self.detr.query_embed.weight, points_fusion=self.args.use_pointsfusion, point_features=point_features)
        else:
            hs, detr_encoder_outs, multiview_encoder_outs = self.detr.transformer(self.detr.input_proj(src), mask, self.detr.query_embed.weight, pos[-1], self.detr.input_proj(src_multiview), mask_multiview, pos_multiview[-1], multiview_fusion=self.args.use_multiviewfusion, points_fusion=self.args.use_pointsfusion, point_features=point_features)
        inst_repr = hs[-1]
        num_nodes = inst_repr.shape[1]

        # Prediction Heads for Object Detection
        if self.args.deformable_detr:
            outputs_class = self.detr.class_embed[-1](hs)
            outputs_coord = self.detr.bbox_embed[-1](hs).sigmoid()
        else:
            outputs_class = self.detr.class_embed(hs)
            outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        # -----------------------------------------------

        det2gt_indices = None
        if self.training:
            detr_outs = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            det2gt_indices = self.detr_matcher(detr_outs, targets)
            gt_rel_pairs = []
            for (ds, gs), t in zip(det2gt_indices, targets):
                ds, gs = ds.to(self.args.device), gs.to(self.args.device)
                gt2det_map = torch.zeros(len(gs)).to(device=ds.device, dtype=ds.dtype)
                gt2det_map[gs] = ds
                gt_rels = gt2det_map[t['relation_map'].sum(-1).nonzero(as_tuple=False)]
                perm = torch.randperm(len(gt_rels))

                gt_rel_pairs.append(gt_rels[perm])
                # if len(gt_rel_pairs[-1]) > self.args.num_hoi_queries:
                #     print(f"imageid={t['image_id']}, gt_relation_count={len(gt_rel_pairs[-1])}")
                #     # check_annotation(samples, targets, rel_num=20, idx=0)

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        pred_rel_exists, pred_rel_pairs, pred_actions = [], [], []
        memory_input, memory_input_mask = features[-1].decompose()
        memory_input_multiview, memory_input_mask_multiview = features_multiview[-1].decompose()
        memory_pos = pos[-1] if self.args.num_feature_levels == 1 else pos[-2]
        memory_pos_multiview = pos_multiview[-1]
        if self.args.relation_feature_map_from == 'backbone':
            relation_feature_map = features[-1]
            relation_feature_map_multiview = features_multiview[-1]
        elif self.args.relation_feature_map_from == 'detr_encoder':
            relation_feature_map = NestedTensor(detr_encoder_outs, memory_input_mask)
            relation_feature_map_multiview = NestedTensor(multiview_encoder_outs, memory_input_mask_multiview)
            memory_input = detr_encoder_outs

        if not self.args.no_interaction_decoder:
            if self.args.backbone == "resnet34":
                memory_input = self.backbone_channel_adapt(memory_input.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
                memory_input_multiview = self.backbone_channel_adapt(
                    memory_input_multiview.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            memory_input = self.memory_input_proj(memory_input)
            memory_input_multiview = self.memory_input_proj(memory_input_multiview)

        for imgid in range(bs):
            # >>>>>>>>>>>> relation proposal <<<<<<<<<<<<<<<
            probs = outputs_class[-1, imgid].softmax(-1)
            inst_scores, inst_labels = probs[:, :-1].max(-1)
            human_instance_ids = torch.logical_and(inst_scores > 0.5, inst_labels <= 10).nonzero(as_tuple=False)
            bg_instance_ids = (probs[:, -1] > 1)
            if self.args.apply_nms_on_detr and not self.training:
                suppress_ids = self.apply_nms(inst_scores, inst_labels, outputs_coord[-1, imgid])
                bg_instance_ids[suppress_ids] = True

            rel_mat = torch.zeros((num_nodes, num_nodes))
            rel_mat[human_instance_ids, ~bg_instance_ids] = 1 # subj is human, obj is not background
            if self.args.dataset_file != 'vcoco': rel_mat.fill_diagonal_(0)
            if self.args.adaptive_relation_query_num:
                if len(rel_mat.nonzero(as_tuple=False)) == 0: rel_mat[0,1] = 1
            else: # ensure enough queries
                if len(rel_mat.nonzero(as_tuple=False)) < self.args.num_hoi_queries:
                    tmp_id = np.random.choice(human_instance_ids.squeeze(1).tolist()) if len(human_instance_ids) > 0 else 0
                    rel_mat[tmp_id] = 1

            if self.training:
                rel_mat[gt_rel_pairs[imgid][:,:1], ~bg_instance_ids] = 1
                rel_mat[gt_rel_pairs[imgid][:,0], gt_rel_pairs[imgid][:, 1]] = 0
                rel_pairs = rel_mat.nonzero(as_tuple=False).to(self.args.device) # neg pairs

                if self.args.use_hard_mining_for_relation_discovery:
                    # hard negative sampling
                    all_pairs = torch.cat([gt_rel_pairs[imgid], rel_pairs], dim=0)
                    gt_pair_count = len(gt_rel_pairs[imgid])
                    all_rel_reps = self.coarse_relation_feature_extractor(all_pairs, relation_feature_map, outputs_coord[-1, imgid].detach(), inst_repr[imgid], obj_label_logits=outputs_class[-1, imgid], idx=imgid, features_multiview=relation_feature_map_multiview)
                    p_relation_exist_logits = self.relation_proposal_mlp(all_rel_reps)

                    gt_inds = torch.arange(gt_pair_count).to(p_relation_exist_logits.device)
                    _, sort_rel_inds = p_relation_exist_logits[gt_pair_count:].squeeze(1).sort(descending=True)
                    # _, sort_rel_inds = torch.cat([inst_scores[all_pairs[:, 1:]], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1)[gt_pair_count:].sort(descending=True)
                    sampled_rel_inds = torch.cat([gt_inds, sort_rel_inds+gt_pair_count])[:self.args.num_hoi_queries]

                    sampled_rel_pairs = all_pairs[sampled_rel_inds]
                    sampled_rel_reps = all_rel_reps[sampled_rel_inds]
                    sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]

                else:
                    # random sampling
                    sampled_neg_inds = torch.randperm(len(rel_pairs))
                    sampled_rel_pairs = torch.cat([gt_rel_pairs[imgid], rel_pairs[sampled_neg_inds]], dim=0)[:self.args.num_hoi_queries]
                    sampled_rel_reps = self.coarse_relation_feature_extractor(sampled_rel_pairs, relation_feature_map, outputs_coord[-1, imgid].detach(), inst_repr[imgid], obj_label_logits=outputs_class[-1, imgid], idx=imgid)
                    sampled_rel_pred_exists = self.relation_proposal_mlp(sampled_rel_reps).squeeze(1)
            else:
                rel_pairs = rel_mat.nonzero(as_tuple=False).to(self.args.device)
                rel_reps = self.coarse_relation_feature_extractor(rel_pairs, relation_feature_map, outputs_coord[-1, imgid].detach(), inst_repr[imgid], obj_label_logits=outputs_class[-1, imgid], idx=imgid, features_multiview=relation_feature_map_multiview)
                p_relation_exist_logits = self.relation_proposal_mlp(rel_reps)

                _, sort_rel_inds = p_relation_exist_logits.squeeze(1).sort(descending=True)
                # _, sort_rel_inds = torch.cat([inst_scores[rel_pairs[:, 1:]], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1).sort(descending=True)
                sampled_rel_inds = sort_rel_inds[:self.args.num_hoi_queries]

                sampled_rel_pairs = rel_pairs[sampled_rel_inds]
                sampled_rel_reps = rel_reps[sampled_rel_inds]
                sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]

            # >>>>>>>>>>>> relation classification <<<<<<<<<<<<<<<
            if self.args.use_simple_pointsfusion and self.args.use_pointsfusion:
                points_feats = point_features[imgid]
                points_feats = self.pointsfeats_proj(points_feats)
                sampled_rel_reps = sampled_rel_reps + torch.matmul(torch.matmul(sampled_rel_reps, points_feats.transpose(0, 1)), points_feats)

            query_reps = self.rel_query_pre_proj(sampled_rel_reps).unsqueeze(1)

            if self.args.no_interaction_decoder:
                outs = query_reps.unsqueeze(0)
            else:
                query_pos_encoding, relation_dependency_encodings, layout_encodings, memory_union_mask, tgt_mask = None, None, None, None, None
                subj_mask, obj_mask, union_mask, _ = self.generate_layout_masks(sampled_rel_pairs, memory_input_mask, outputs_coord[-1, imgid], idx=imgid)
                if self.args.use_relation_tgt_mask:
                    tgt_mask = (torch.diag(sampled_rel_pred_exists) != 0)
                    attend_ids = sampled_rel_pred_exists.sort(descending=True)[1][:self.args.use_relation_tgt_mask_attend_topk]
                    tgt_mask[:, attend_ids] = True
                    tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
                if self.args.use_query_fourier_encoding:
                    query_coords = self.fourier_feature_embedding(outputs_coord[-1, imgid][sampled_rel_pairs].view(len(sampled_rel_pairs), 8, 1)) / np.sqrt(self.args.hidden_dim/2)
                    query_pos_encoding = self.fourier_mlp(torch.cat([torch.cos(query_coords), torch.sin(query_coords)], dim=-1)).view(len(sampled_rel_pairs), -1).unsqueeze(1)
                if self.args.use_relation_dependency_encoding:
                    dependency_map = torch.zeros((len(sampled_rel_pairs), len(sampled_rel_pairs))).to(sampled_rel_reps.device).long() # independent: 0
                    dependency_map[sampled_rel_pairs[:, 0].unsqueeze(1) == sampled_rel_pairs[:, 0].unsqueeze(0)] = 1 # same_subj: 1
                    dependency_map[sampled_rel_pairs[:, 1].unsqueeze(1) == sampled_rel_pairs[:, 1].unsqueeze(0)] = 2 # same_obj: 2
                    dependency_map[sampled_rel_pairs[:, 0].unsqueeze(1) == sampled_rel_pairs[:, 1].unsqueeze(0)] = 3 # subj=obj: 3
                    dependency_map[sampled_rel_pairs[:, 1].unsqueeze(1) == sampled_rel_pairs[:, 0].unsqueeze(0)] = 4 # obj=subj: 4
                    dependency_map.fill_diagonal_(5) # self: 5
                    relation_dependency_encodings = self.relation_dependency_embeddings(dependency_map)
                    relation_dependency_encodings = self.relation_dependency_content_aware_mapping(
                        torch.cat([query_reps.permute(1,0,2).expand(*relation_dependency_encodings.shape), relation_dependency_encodings], dim=-1)
                    ).unsqueeze(2) # (#query, #query, batch size, dim)
                if self.args.use_prior:
                    sampled_rel_class_logits = torch.cat(
                        [outputs_class[-1, imgid][:, :-1][sampled_rel_pairs[:, 0]].unsqueeze(-1),
                         outputs_class[-1, imgid][:, :-1][sampled_rel_pairs[:, 1]].unsqueeze(-1)], dim=-1)
                    sampled_rel_class = sampled_rel_class_logits.argmax(-2)
                    prior_map = torch.zeros((len(sampled_rel_pairs), 1)).to(
                        sampled_rel_reps.device).long()
                    prior_map[sampled_rel_class[:, 0] <= 4] = 1
                    prior_map[(sampled_rel_class[:, 0] >= 6) * (sampled_rel_class[:, 0] <= 8) * (
                                sampled_rel_class[:, 1] >= 6) * (sampled_rel_class[:, 1] <= 8)] = 2
                    prior_map[(sampled_rel_class[:, 0] >= 6) * (sampled_rel_class[:, 0] <= 7) * (
                                sampled_rel_class[:, 1] == 5)] = 3
                    prior_map[(sampled_rel_class[:, 0] == 8) * (sampled_rel_class[:, 1] == 5)] = 4
                    prior_map[(sampled_rel_class[:, 0] >= 6) * (sampled_rel_class[:, 0] <= 8) * (
                                sampled_rel_class[:, 1] == 4)] = 5
                    prior_map[(sampled_rel_class[:, 0] >= 6) * (sampled_rel_class[:, 0] <= 7) * (
                                sampled_rel_class[:, 1] == 1)] = 6
                    prior_encodings = self.prior_embeddings(prior_map)
                    query_pos_encoding = prior_encodings

                if self.args.use_memory_union_mask:
                    memory_union_mask = union_mask.flatten(1)
                if self.args.use_memory_layout_encoding:
                    layout_map = (~union_mask).long() + (~memory_input_mask[imgid:imgid+1]).long() + (~subj_mask).long() + (~obj_mask).long()*2
                    # plt.imshow(role_map[0].cpu().numpy(), cmap=plt.cm.hot_r); plt.colorbar(); plt.show()
                    layout_encodings = self.layout_embeddings(layout_map)
                    layout_encodings = self.layout_content_aware_mapping(
                        torch.cat([memory_input[imgid:imgid+1].permute(0,2,3,1).expand(*layout_encodings.shape), layout_encodings], dim=-1)
                    ).flatten(start_dim=1, end_dim=2).unsqueeze(2) # (#query, #memory, batch size, dim)


                if self.args.use_multiviewfusion_last_all:
                    memory_input_last = torch.cat([memory_input[imgid:imgid + 1].flatten(2).permute(2, 0, 1),
                                              memory_input_multiview.split(3, dim=0)[imgid].flatten(2).permute(0, 2, 1).flatten(0, 1).unsqueeze(1)], dim=0)
                    memory_input_mask_last = torch.cat([memory_input_mask[imgid:imgid + 1].flatten(1),
                                                   memory_input_mask_multiview.split(3, dim=0)[imgid].flatten(
                                                       1).flatten(0).unsqueeze(0)], dim=1)
                    memory_pos_last = torch.cat([memory_pos[imgid:imgid + 1].flatten(2).permute(2, 0, 1),
                                            memory_pos_multiview.split(3, dim=0)[imgid].flatten(2).permute(0, 2, 1).flatten(0, 1).unsqueeze(1)], dim=0)
                    layout_encodings_last = torch.cat([layout_encodings, torch.zeros(layout_encodings.shape[0], layout_encodings.shape[1]*3, layout_encodings.shape[2], layout_encodings.shape[3],).to(self.args.device)], dim=1)

                    outs = self.interaction_decoder(tgt=query_reps,
                                                    tgt_mask=tgt_mask,
                                                    query_pos=query_pos_encoding,
                                                    query_structure_encoding=relation_dependency_encodings, # inter-ineraction semantic structure
                                                    memory=memory_input_last,
                                                    memory_key_padding_mask=memory_input_mask_last,
                                                    memory_mask=memory_union_mask,
                                                    pos=memory_pos_last,
                                                    memory_role_embedding=layout_encodings_last) #  intra-ineraction spatial structure


                elif self.args.use_multiviewfusion_last_view2:
                    memory_input_last = torch.cat([memory_input[imgid:imgid + 1].flatten(2).permute(2, 0, 1),
                                              memory_input_multiview[2::3][imgid:imgid + 1].flatten(2).permute(0, 2, 1).flatten(0, 1).unsqueeze(1),memory_input_multiview[0::3][imgid:imgid + 1].flatten(2).permute(0, 2, 1).flatten(0, 1).unsqueeze(1)], dim=0)
                    memory_input_mask_last = torch.cat([memory_input_mask[imgid:imgid + 1].flatten(1),
                                                   memory_input_mask_multiview[2::3][imgid:imgid + 1].flatten(
                                                       1).flatten(0).unsqueeze(0), memory_input_mask_multiview[0::3][imgid:imgid + 1].flatten(
                                                       1).flatten(0).unsqueeze(0)], dim=1)
                    memory_pos_last = torch.cat([memory_pos[imgid:imgid + 1].flatten(2).permute(2, 0, 1),
                                            memory_pos_multiview[2::3][imgid:imgid + 1].flatten(2).permute(0, 2, 1).flatten(0, 1).unsqueeze(1), memory_pos_multiview[0::3][imgid:imgid + 1].flatten(2).permute(0, 2, 1).flatten(0, 1).unsqueeze(1)], dim=0)
                    layout_encodings_last = torch.cat([layout_encodings, torch.zeros(layout_encodings.shape[0], layout_encodings.shape[1]*2, layout_encodings.shape[2], layout_encodings.shape[3],).to(self.args.device)], dim=1)

                    outs = self.interaction_decoder(tgt=query_reps,
                                                    tgt_mask=tgt_mask,
                                                    query_pos=query_pos_encoding,
                                                    query_structure_encoding=relation_dependency_encodings, # inter-ineraction semantic structure
                                                    memory=memory_input_last,
                                                    memory_key_padding_mask=memory_input_mask_last,
                                                    memory_mask=memory_union_mask,
                                                    pos=memory_pos_last,
                                                    memory_role_embedding=layout_encodings_last) #  intra-ineraction spatial structure

                elif self.args.use_multiviewfusion_last:
                    memory_input_last = torch.cat([memory_input[imgid:imgid + 1].flatten(2).permute(2, 0, 1),
                                              memory_input_multiview[2::3][imgid:imgid + 1].flatten(2).permute(0, 2, 1).flatten(0, 1).unsqueeze(1)], dim=0)
                    memory_input_mask_last = torch.cat([memory_input_mask[imgid:imgid + 1].flatten(1),
                                                   memory_input_mask_multiview[2::3][imgid:imgid + 1].flatten(
                                                       1).flatten(0).unsqueeze(0)], dim=1)
                    memory_pos_last = torch.cat([memory_pos[imgid:imgid + 1].flatten(2).permute(2, 0, 1),
                                            memory_pos_multiview[2::3][imgid:imgid + 1].flatten(2).permute(0, 2, 1).flatten(0, 1).unsqueeze(1)], dim=0)
                    layout_encodings_last = torch.cat([layout_encodings, torch.zeros(layout_encodings.shape[0], layout_encodings.shape[1]*1, layout_encodings.shape[2], layout_encodings.shape[3],).to(self.args.device)], dim=1)

                    outs = self.interaction_decoder(tgt=query_reps,
                                                    tgt_mask=tgt_mask,
                                                    query_pos=query_pos_encoding,
                                                    query_structure_encoding=relation_dependency_encodings, # inter-ineraction semantic structure
                                                    memory=memory_input_last,
                                                    memory_key_padding_mask=memory_input_mask_last,
                                                    memory_mask=memory_union_mask,
                                                    pos=memory_pos_last,
                                                    memory_role_embedding=layout_encodings_last) #  intra-ineraction spatial structure

                else:
                    outs = self.interaction_decoder(tgt=query_reps,
                                                    tgt_mask=tgt_mask,
                                                    query_pos=query_pos_encoding,
                                                    query_structure_encoding=relation_dependency_encodings, # inter-ineraction semantic structure
                                                    memory=memory_input[imgid:imgid+1].flatten(2).permute(2,0,1),
                                                    memory_key_padding_mask=memory_input_mask[imgid:imgid+1].flatten(1),
                                                    memory_mask=memory_union_mask,
                                                    pos=memory_pos[imgid:imgid+1].flatten(2).permute(2, 0, 1),
                                                    memory_role_embedding=layout_encodings) #  intra-ineraction spatial structure

            action_logits = self.action_embed(outs)

            pred_rel_pairs.append(sampled_rel_pairs)
            pred_actions.append(action_logits)
            pred_rel_exists.append(sampled_rel_pred_exists)

        hoi_recognition_time = time.time() - start_time
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_rel_pairs": pred_rel_pairs,
            "pred_actions": [p[-1].squeeze(1) for p in pred_actions],
            "pred_action_exists": pred_rel_exists,
            "det2gt_indices": det2gt_indices,
            "hoi_recognition_time": hoi_recognition_time,
        }
        if self.args.hoi_aux_loss: out['hoi_aux_outputs'] = self._set_hoi_aux_loss(pred_actions)
        if self.args.train_detr and self.args.aux_loss: out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_hoi_aux_loss(self, pred_actions):
        return [{'pred_actions': [p[l].squeeze(1) for p in pred_actions]} for l in range(self.args.hoi_dec_layers - 1)]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": l, "pred_boxes": b} for l, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # merge boxes (NMS)
    def apply_nms(self, inst_scores, inst_labels, cxcywh_boxes, threshold=0.7):
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(cxcywh_boxes)
        box_areas = (xyxy_boxes[:, 2:] - xyxy_boxes[:, :2]).prod(-1)
        box_area_sum = box_areas.unsqueeze(1) + box_areas.unsqueeze(0)

        union_boxes = torch.cat([torch.min(xyxy_boxes.unsqueeze(1)[:, :, :2], xyxy_boxes.unsqueeze(0)[:, :, :2]),
                                 torch.max(xyxy_boxes.unsqueeze(1)[:, :, 2:], xyxy_boxes.unsqueeze(0)[:, :, 2:])], dim=-1)
        union_area = (union_boxes[:,:,2:] - union_boxes[:,:,:2]).prod(-1)
        iou = torch.clamp(box_area_sum - union_area, min=0) / union_area
        box_match_mat = torch.logical_and(iou > threshold, inst_labels.unsqueeze(1) == inst_labels.unsqueeze(0))

        suppress_ids = []
        for box_match in box_match_mat:
            group_ids = box_match.nonzero(as_tuple=False).squeeze(1)
            if len(group_ids) > 1:
                max_score_inst_id = group_ids[inst_scores[group_ids].argmax()]
                bg_ids = group_ids[group_ids!=max_score_inst_id]
                suppress_ids.append(bg_ids)
                box_match_mat[:, bg_ids] = False
        if len(suppress_ids) > 0:
            suppress_ids = torch.cat(suppress_ids, dim=0)
        return suppress_ids

    def generate_layout_masks(self, rel_pairs, feature_masks, boxes, idx):
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(0, 1)
        head_boxes = xyxy_boxes[rel_pairs[:, 0]]
        tail_boxes = xyxy_boxes[rel_pairs[:, 1]]
        union_boxes = torch.cat([
            torch.min(head_boxes[:,:2], tail_boxes[:,:2]),
            torch.max(head_boxes[:,2:], tail_boxes[:,2:])
        ], dim=1)

        h, w = (~feature_masks[idx]).nonzero(as_tuple=False).max(dim=0)[0] + 1 # mask: image area=False, pad area=True
        scaled_head_boxes = head_boxes * torch.tensor([w,h,w,h]).to(device=head_boxes.device, dtype=head_boxes.dtype).unsqueeze(0)
        scaled_tail_boxes = tail_boxes * torch.tensor([w,h,w,h]).to(device=tail_boxes.device, dtype=tail_boxes.dtype).unsqueeze(0)
        scaled_union_boxes = union_boxes * torch.tensor([w,h,w,h]).to(device=union_boxes.device, dtype=union_boxes.dtype).unsqueeze(0)
        bound_upper_inds = (torch.tensor([w,h,w,h])-1).unsqueeze(0).float().to(feature_masks.device)
        rounded_head_boxes = torch.min(scaled_head_boxes.round(), bound_upper_inds).int()
        rounded_tail_boxes = torch.min(scaled_tail_boxes.round(), bound_upper_inds).int()
        rounded_union_boxes = torch.min(scaled_union_boxes.round(), bound_upper_inds).int()

        role_embeddings = None
        # build masks: hit region=False, other region=True
        rel_head_mask = torch.ones_like(feature_masks[idx]).unsqueeze(0).repeat((len(rel_pairs), 1, 1))
        rel_tail_mask = torch.ones_like(feature_masks[idx]).unsqueeze(0).repeat((len(rel_pairs), 1, 1))
        rel_union_mask = torch.ones_like(feature_masks[idx]).unsqueeze(0).repeat((len(rel_pairs), 1, 1))
        for rid in range(len(rel_union_mask)):
            rel_head_mask[rid, rounded_head_boxes[rid,1]:rounded_head_boxes[rid,3]+1,
                               rounded_head_boxes[rid,0]:rounded_head_boxes[rid,2]+1] = False
            rel_tail_mask[rid, rounded_tail_boxes[rid,1]:rounded_tail_boxes[rid,3]+1,
                               rounded_tail_boxes[rid,0]:rounded_tail_boxes[rid,2]+1] = False
            rel_union_mask[rid, rounded_union_boxes[rid,1]:rounded_union_boxes[rid,3]+1,
                                rounded_union_boxes[rid,0]:rounded_union_boxes[rid,2]+1] = False

        return rel_head_mask, rel_tail_mask, rel_union_mask, role_embeddings

class STIPCriterion(nn.Module):
    """ This class computes the loss for s2former.
    1. proposal loss
    2. relation classification loss
    """
    def __init__(self, args, matcher):
        super().__init__()
        self.args = args
        self.matcher = matcher
        self.weight_dict = {
            'loss_proposal': args.proposal_loss_coef,
            'loss_act': args.action_loss_coef
        }
        if args.hoi_aux_loss:
            for i in range(args.hoi_dec_layers - 1):
                self.weight_dict.update({f'loss_act_{i}': self.weight_dict['loss_act']})

        if args.dataset_file == 'vcoco':
            self.invalid_ids = args.invalid_ids
            self.valid_ids = args.valid_ids
        elif args.dataset_file == 'hico-det':
            self.invalid_ids = []
            self.valid_ids = list(range(self.args.num_actions))
            self.hico_valid_obj_ids = torch.tensor(self.args.valid_obj_ids)
        elif args.dataset_file == 'or':
            self.invalid_ids = []
            self.valid_ids = list(range(self.args.num_actions))

        if args.train_detr:
            self.num_classes = args.num_classes
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = args.eos_coef
            self.register_buffer('empty_weight', empty_weight)

            self.detr_losses = ['labels', 'boxes', 'cardinality']
            det_weights = {'loss_ce': 1 * args.finetune_detr_weight, 'loss_bbox': args.bbox_loss_coef * args.finetune_detr_weight, 'loss_giou': args.giou_loss_coef * args.finetune_detr_weight}
            if args.aux_loss:
                aux_weights = {}
                for i in range(args.dec_layers - 1):
                    aux_weights.update({k + f'_{i}': v for k, v in det_weights.items()})
                det_weights.update(aux_weights)
            self.weight_dict.update(det_weights)

    #######################################################################################################################
    # * DETR Losses
    #######################################################################################################################
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, log=False):
        # instance matching
        if outputs['det2gt_indices'] is None:
            outputs_without_aux = {k: v for k, v in outputs.items() if (k != 'aux_outputs' and k != 'hoi_aux_outputs')}
            indices = self.matcher(outputs_without_aux, targets)
        else:
            indices = outputs['det2gt_indices']

        # generate relation targets
        all_rel_pair_targets = []
        for imgid, (tgt, (det_idxs, gtbox_idxs)) in enumerate(zip(targets, indices)):
            det2gt_map = {int(d): int(g) for d, g in zip(det_idxs, gtbox_idxs)}
            gt_relation_map = tgt['relation_map']
            rel_pairs = outputs['pred_rel_pairs'][imgid]
            rel_pair_targets = torch.zeros((len(rel_pairs), gt_relation_map.shape[-1])).to(gt_relation_map.device)
            for idx, rel in enumerate(rel_pairs):
                if (int(rel[0]) in det2gt_map) and (int(rel[1]) in det2gt_map):
                    rel_pair_targets[idx] = gt_relation_map[det2gt_map[int(rel[0])], det2gt_map[int(rel[1])]]
            all_rel_pair_targets.append(rel_pair_targets)
        all_rel_pair_targets = torch.cat(all_rel_pair_targets, dim=0)

        prior_verb_label_mask = None
        if self.args.dataset_file == 'hico-det':
            # no_interaction_id = self.args.action_names.index('no_interaction')
            # rel_proposal_targets = (all_rel_pair_targets[..., self.valid_ids].sum(-1) - all_rel_pair_targets[..., no_interaction_id] > 0).float()
            rel_proposal_targets = (all_rel_pair_targets[..., self.valid_ids].sum(-1) > 0).float()
            if self.args.use_prior_verb_label_mask:
                pred_obj_labels = outputs['pred_logits'][:,:,self.args.valid_obj_ids].argmax(-1)
                tail_obj_ids = [p[:,1] for p in outputs['pred_rel_pairs']]
                tail_obj_labels = torch.cat([l[id] for l, id in zip(pred_obj_labels, tail_obj_ids)])
                prior_verb_label_mask = self.args.correct_mat.transpose(0,1)[tail_obj_labels]
        else:
            rel_proposal_targets = (all_rel_pair_targets[..., self.valid_ids].sum(-1) > 0).float()

        loss_proposal = self.proposal_loss(torch.cat(outputs['pred_action_exists'], dim=0), rel_proposal_targets)
        loss_action = self.action_loss(torch.cat(outputs['pred_actions'], dim=0)[..., self.valid_ids], all_rel_pair_targets[..., self.valid_ids], prior_verb_label_mask)

        loss_dict = {'loss_proposal': loss_proposal, 'loss_act': loss_action}
        if 'hoi_aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['hoi_aux_outputs']):
                aux_loss = {
                    f'loss_act_{i}': self.action_loss(torch.cat(aux_outputs['pred_actions'], dim=0)[..., self.valid_ids], all_rel_pair_targets[..., self.valid_ids], prior_verb_label_mask)
                }
                loss_dict.update(aux_loss)

        # jointly train objects and relation decoder
        if self.args.train_detr:
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.detr_losses:
                loss_dict.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.detr_losses:
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        loss_dict.update(l_dict)

        return loss_dict

    def proposal_loss(self, inputs, targets):
        # loss = focal_loss(inputs, targets, gamma=self.args.proposal_focal_loss_gamma, alpha=self.args.proposal_focal_loss_alpha)

        ## conventional BCE
        # loss_bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss = loss_bce[targets<0.5].mean() # neg loss
        # if targets.sum() > 0:
        #     loss += loss_bce[targets>0.5].mean() # pos loss

        # focal loss to balance positive/negative
        probs = inputs.sigmoid()
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()
        pos_loss = torch.log(probs) * torch.pow(1 - probs, self.args.proposal_focal_loss_gamma) * pos_inds
        neg_loss = torch.log(1 - probs) * torch.pow(probs, self.args.proposal_focal_loss_gamma) * neg_inds
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # normalize
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

    def action_loss(self, inputs, targets, prior_verb_label_mask=None):
        # loss = focal_loss(inputs, targets, gamma=self.args.action_focal_loss_gamma, alpha=self.args.action_focal_loss_alpha, prior_verb_label_mask=prior_verb_label_mask)
        probs = inputs.sigmoid()


        weights =  torch.tensor([1.7361e-03, 1.6393e-02,3.3003e-03, 1.9543e-05, 8.4034e-03, 1.6129e-02,6.6225e-03,3.8226e-04,2.8694e-04,3.2680e-03,7.6104e-04,8.0000e-03,1.2195e-02, 9.5785e-04])
        # focal loss to balance positive/negative
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()
        pos_loss = torch.log(probs) * torch.pow(1 - probs, self.args.action_focal_loss_gamma) * pos_inds
        neg_loss = torch.log(1 - probs) * torch.pow(probs, self.args.action_focal_loss_gamma) * neg_inds
        if prior_verb_label_mask is not None: # mask invalid predictions
            pos_loss = pos_loss * prior_verb_label_mask
            neg_loss = neg_loss * prior_verb_label_mask
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # normalize
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss

class STIPPostProcess(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args

    @torch.no_grad()
    def forward(self, outputs, target_sizes=None, threshold=0, dataset='coco'):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        if target_sizes is None:
            target_sizes = torch.zeros([len(out_logits), 2]).to(out_logits.device).type(torch.long)
            target_sizes[:, 0], target_sizes[:, 1] = 1536, 2048
            # target_sizes = torch.tensor([[1536, 2048]]).type(torch.long).to(out_logits.device)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # for relationship post-processing
        h_indices = [p[:, 0] for p in outputs['pred_rel_pairs']]
        o_indices = [p[:, 1] for p in outputs['pred_rel_pairs']]

        _valid_obj_ids = self.args.valid_obj_ids + [self.args.valid_obj_ids[-1]+1]
        out_obj_logits = outputs['pred_logits'][..., _valid_obj_ids]
        # tail classification score
        obj_scores, obj_labels = [], []
        sub_scores, sub_labels = [], []
        for o_ids, lgts in zip(o_indices, out_obj_logits):
            img_obj_scores, img_obj_labels = F.softmax(lgts[o_ids], -1)[..., :-1].max(-1)
            obj_scores.append(img_obj_scores)
            obj_labels.append(img_obj_labels)
        for h_ids, lgts in zip(h_indices, out_obj_logits):
            img_sub_scores, img_sub_labels = F.softmax(lgts[h_ids], -1)[..., :-1].max(-1)
            sub_scores.append(img_sub_scores)
            sub_labels.append(img_sub_labels)
        # actions
        out_verb_logits = outputs['pred_actions']
        verb_scores = [l.sigmoid() for l in out_verb_logits]
        verb_labels = [torch.argmax(v, -1) for v in verb_scores]
        # verb_scores = out_verb_logits.sigmoid() * outputs['pred_action_exists'].sigmoid().unsqueeze(-1) # interactiveness

        # accumulate results (iterate through interaction queries)
        results = []
        for batch_idx, (os, sl, ol, vs, box, h_idx, o_idx, vl, ss) in enumerate(zip(obj_scores, sub_labels, obj_labels, verb_scores, boxes, h_indices, o_indices, verb_labels, sub_scores)):
            # label
            # sl = torch.full_like(ol, 0) # self.subject_category_id = 0 in HICO-DET
            l = torch.cat((sl, ol))
            # boxes
            sb = box[h_idx, :]
            ob = box[o_idx, :]
            b = torch.cat((sb, ob))

            vs = vs * vs * os.unsqueeze(1) * ss.unsqueeze(1)
            ids = torch.arange(b.shape[0])

            ts = [vs[a][vl[a]] for a in range(len(vs))]
            res_dict = {
                'labels': l.to('cpu'),
                'boxes': b.to('cpu'),
                'verb_scores': vs.to('cpu'),
                'sub_ids': ids[:ids.shape[0] // 2],
                'obj_ids': ids[ids.shape[0] // 2:],
                'hoi_recognition_time': outputs['hoi_recognition_time'],
                'orig_size': torch.tensor([img_h[batch_idx], img_w[batch_idx]])
            }
            triplet = []
            for i in range(len(res_dict['sub_ids'])):
                triplet.append([res_dict['labels'][res_dict['sub_ids'][i]], res_dict['labels'][res_dict['obj_ids']][i], vl[i]])

            total_scores = torch.tensor(ts)
            ranked_orders = np.argsort(-total_scores, 0)
            ranked_orders = [int(k) for k in ranked_orders]
            triplet = torch.tensor(triplet)
            triplet = triplet[ranked_orders, :]
            total_scores = total_scores[ranked_orders]

            res_dict["triplet"] = triplet
            res_dict["ranked_scores"] = total_scores
            results.append(res_dict)

        return results

class RelationFeatureExtractor(nn.Module):
    def __init__(self, args, in_channels, resolution=5, out_dim=1024):
        super(RelationFeatureExtractor, self).__init__()
        self.args = args
        self.resolution = resolution

        # head & tail feature (base feature)
        instr_hidden_dim = self.args.hidden_dim
        fusion_dim = instr_hidden_dim*2

        # spatial feature
        if args.use_spatial_feature:
            spatial_in_dim, spatial_out_dim = 8, 64
            self.spatial_proj = make_fc(spatial_in_dim, spatial_out_dim)
            fusion_dim += spatial_out_dim

        # tail semantic feature
        if args.use_tail_semantic_feature:
            semantic_dim = 300
            self.label_embedding = nn.Embedding(self.args.num_classes+1, semantic_dim)
            if self.args.use_head_semantic_feature:
                fusion_dim += semantic_dim * 2
            else:
                fusion_dim += semantic_dim

        # union feature
        if args.use_union_feature:
            out_ch, union_out_dim = 256, 256
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_ch, kernel_size=1),
                nn.ReLU(inplace=True),
            ) # reduce channel size before pooling
            self.visual_proj = make_fc(out_ch * (resolution**2), union_out_dim)
            fusion_dim += union_out_dim

        if args.use_view6:
            out_ch, union_out_dim = 256, 256
            self.input_proj_view6 = nn.Sequential(
                nn.Conv2d(in_channels, out_ch, kernel_size=1),
                nn.ReLU(inplace=True),
            ) # reduce channel size before pooling
            self.visual_proj_view6 = make_fc(out_ch * (resolution**2), union_out_dim)
            fusion_dim += union_out_dim

        unionvideo_decoder_layer = TransformerDecoderLayer_multiview(self.args.hidden_dim, self.args.hoi_nheads)
        unionvideo_decoder_norm = nn.LayerNorm(self.args.hidden_dim)
        self.unionvideo_attention = TransformerDecoder(unionvideo_decoder_layer, 2, unionvideo_decoder_norm, return_intermediate=False)

        # fusion
        self.fusion_fc = nn.Sequential(
            make_fc(fusion_dim, out_dim), nn.ReLU(),
            make_fc(out_dim, out_dim), nn.ReLU()
        )

    def forward(self, rel_pairs, features, boxes, inst_reprs, idx, obj_label_logits=None, features_multiview=None, features_video=None):
        """pool feature for boxes on one image
            features: dxhxw
            boxes: Nx4 (cx_cy_wh, nomalized to 0-1)
            rel_pairs: Nx2
        """
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(0, 1)
        head_boxes = xyxy_boxes[rel_pairs[:, 0]]
        tail_boxes = xyxy_boxes[rel_pairs[:, 1]]
        union_boxes = torch.cat([
            torch.min(head_boxes[:,:2], tail_boxes[:,:2]),
            torch.max(head_boxes[:,2:], tail_boxes[:,2:])
        ], dim=1)
        view6_boxes = torch.zeros_like(union_boxes)
        view6_boxes[:, :2] = 0
        view6_boxes[:, 2:] = 1

        # head & tail features
        head_feats = inst_reprs[rel_pairs[:, 0]]
        tail_feats = inst_reprs[rel_pairs[:, 1]]
        tail_feats[rel_pairs[:, 0]==rel_pairs[:, 1]] = 0 # set to 0 when head==tail for VCOCO (i.e., tail overlapped)
        head_feats[rel_pairs[:, 0] == rel_pairs[:, 1]] = 0  # set to 0 when head==tail for VCOCO (i.e., tail overlapped)

        relation_feats = torch.cat([head_feats, tail_feats], dim=-1)

        # spatial layout feats
        if self.args.use_spatial_feature:
            box_layout_feats = self.extract_spatial_layout_feats(xyxy_boxes)
            rel_spatial_feats = self.spatial_proj(box_layout_feats[rel_pairs[:,0], rel_pairs[:,1]])
            relation_feats = torch.cat([relation_feats, rel_spatial_feats], dim=-1)

        # semantic feature
        if self.args.use_tail_semantic_feature and self.args.use_head_semantic_feature:
            semantic_feats_tail = (obj_label_logits.softmax(-1) @ self.label_embedding.weight)[rel_pairs[:, 1]]
            semantic_feats_head = (obj_label_logits.softmax(-1) @ self.label_embedding.weight)[rel_pairs[:, 0]]
            relation_feats = torch.cat([relation_feats, semantic_feats_head, semantic_feats_tail], dim=-1)
        else:
            semantic_feats_tail = (obj_label_logits.softmax(-1) @ self.label_embedding.weight)[rel_pairs[:, 1]]
            relation_feats = torch.cat([relation_feats, semantic_feats_tail], dim=-1)

        # union feature
        if self.args.use_union_feature:
            # H, W = features.tensors.shape[-2:] # stacked image size
            h, w = (~features.mask[idx]).nonzero(as_tuple=False).max(dim=0)[0] + 1 # mask: image area=False, pad area=True
            proj_feature = self.input_proj(features.tensors[idx:idx+1])
            proj_feature_video = self.input_proj(features_video.tensors.split(2)[idx])
            scaled_union_boxes = torch.cat(
                [
                    torch.zeros((len(union_boxes),1)).to(device=union_boxes.device),
                    union_boxes * torch.tensor([w,h,w,h]).to(device=union_boxes.device, dtype=union_boxes.dtype).unsqueeze(0),
                ], dim=-1
            )
            union_visual_feats = roi_align(proj_feature, scaled_union_boxes, output_size=self.resolution, sampling_ratio=2)
            union_visual_feats = self.visual_proj(union_visual_feats.flatten(start_dim=1))
            union_visual_feats_video = [roi_align(proj_feature_video.split(1)[0], scaled_union_boxes, output_size=self.resolution, sampling_ratio=2), roi_align(proj_feature_video.split(1)[1], scaled_union_boxes, output_size=self.resolution, sampling_ratio=2)]
            union_visual_feats_video = [self.visual_proj(k.flatten(start_dim=1)) for k in union_visual_feats_video]
            union_visual_feats_video = torch.cat(union_visual_feats_video, dim=0)
            union_visual_feats = self.unionvideo_attention(union_visual_feats.unsqueeze(1), union_visual_feats_video.unsqueeze(1))[0].squeeze(1)
            relation_feats = torch.cat([relation_feats, union_visual_feats], dim=-1)


            # relation_feats = torch.cat([relation_feats, union_visual_feats_video], dim=-1)

        if self.args.use_view6:
            # H, W = features.tensors.shape[-2:] # stacked image size
            h, w = (~features_multiview.mask[2::3][idx]).nonzero(as_tuple=False).max(dim=0)[0] + 1 # mask: image area=False, pad area=True
            proj_feature_view6 = self.input_proj_view6(features_multiview.tensors[2::3][idx:idx+1])
            scaled_view6_boxes = torch.cat(
                [
                    torch.zeros((len(view6_boxes),1)).to(device=union_boxes.device),
                    view6_boxes * torch.tensor([w,h,w,h]).to(device=union_boxes.device, dtype=union_boxes.dtype).unsqueeze(0),
                ], dim=-1
            )
            view6_visual_feats = roi_align(proj_feature_view6, scaled_view6_boxes, output_size=self.resolution, sampling_ratio=2)
            view6_visual_feats = self.visual_proj_view6(view6_visual_feats.flatten(start_dim=1))
            relation_feats = torch.cat([relation_feats, view6_visual_feats], dim=-1)

        x = self.fusion_fc(relation_feats)
        return x

    def extract_spatial_layout_feats(self, xyxy_boxes):
        box_center = torch.stack([(xyxy_boxes[:, 0] + xyxy_boxes[:, 2]) / 2, (xyxy_boxes[:, 1] + xyxy_boxes[:, 3]) / 2], dim=1)
        dxdy = box_center.unsqueeze(1) - box_center.unsqueeze(0) # distances
        theta = (torch.atan2(dxdy[...,1], dxdy[...,0]) / np.pi).unsqueeze(-1)
        dis = dxdy.norm(dim=-1, keepdim=True)

        box_area = (xyxy_boxes[:, 2:] - xyxy_boxes[:, :2]).prod(dim=1) # areas
        intersec_lt = torch.max(xyxy_boxes.unsqueeze(1)[...,:2], xyxy_boxes.unsqueeze(0)[...,:2])
        intersec_rb = torch.min(xyxy_boxes.unsqueeze(1)[...,2:], xyxy_boxes.unsqueeze(0)[...,2:])
        overlap = (intersec_rb - intersec_lt).clamp(min=0).prod(dim=-1, keepdim=True)
        union_lt = torch.min(xyxy_boxes.unsqueeze(1)[...,:2], xyxy_boxes.unsqueeze(0)[...,:2])
        union_rb = torch.max(xyxy_boxes.unsqueeze(1)[...,2:], xyxy_boxes.unsqueeze(0)[...,2:])
        union = (union_rb - union_lt).clamp(min=0).prod(dim=-1, keepdim=True)
        spatial_feats = torch.cat([
            dxdy, dis, theta, # dx, dy, distance, theta
            overlap, union, box_area[:,None,None].expand(*union.shape), box_area[None,:,None].expand(*union.shape) # overlap, union, subj, obj
        ], dim=-1)
        return spatial_feats

# conventional focal loss to balance hard/easy
def focal_loss(blogits, target_classes, alpha=0.5, gamma=2, prior_verb_label_mask=None, class_weights=None):
    probs = blogits.sigmoid() # prob(positive)
    loss_bce = F.binary_cross_entropy_with_logits(blogits, target_classes, reduction='none', weight=class_weights)
    p_t = probs * target_classes + (1 - probs) * (1 - target_classes)
    loss_bce = ((1-p_t)**gamma * loss_bce)

    alpha_t = alpha * target_classes + (1 - alpha) * (1 - target_classes)
    loss_focal = alpha_t * loss_bce

    if prior_verb_label_mask is not None:
        loss_focal = loss_focal * prior_verb_label_mask

    loss = loss_focal.sum() / max(target_classes.sum(), 1)
    return loss

def make_fc(dim_in, hidden_dim, a=1):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
        a: negative slope
    '''
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=a)
    nn.init.constant_(fc.bias, 0)
    return fc


def make_conv3x3(
    in_channels,
    out_channels,
    padding=1,
    dilation=1,
    stride=1,
    kaiming_init=True
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
        nn.init.constant_(conv.bias, 0)
    return conv
