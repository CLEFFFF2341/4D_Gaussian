# Copyright (C) 2023, Inria; GRAPHDECO research group.
# Non-commercial research/evaluation terms: original MoRGS LICENSE.md.
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Six-attribute adapter for the MoRGS online engineering backbone.
from tracking import TrackedModel
from scene.gaussian_model import *
class CleanModel(TrackedModel):

    def __init__(self, sh_degree, latent_args, model_args, **kwargs):
        assert latent_args.param_names == ['xyz', 'f_dc', 'f_rest', 'sc', 'rot', 'op']
        for flag in ('morgs_flow_supervision', 'morgs_motion_offset', 'morgs_motion_confidence'):
            assert getattr(model_args, flag) is False, 'Clean entry cannot enable MORGS module: ' + flag
        super().__init__(sh_degree, latent_args, model_args, **kwargs)
        self.baseline_revision = 'morgs_removed_three_v1'
        for key in ('flow', 'motion_label'):
            self.feature_dims.pop(key)
        for key in ('mask_flow', 'mask_motion_label', 'frz_flow', 'frz_motion_label', 'motion_label_compressed_threshold'):
            delattr(self, key)

    @property
    def get_decoded_atts(self):
        return OrderedDict({'xyz': self._xyz, 'f_dc': self._features_dc, 'f_rest': self._features_rest, 'sc': self._scaling, 'rot': self._rotation, 'op': self._opacity})

    @property
    def get_masks(self):
        return OrderedDict({'xyz': self.mask_xyz, 'f_dc': self.mask_features_dc, 'f_rest': self.mask_features_rest, 'sc': self.mask_scaling, 'rot': self.mask_rotation, 'op': self.mask_opacity})

    @property
    def get_frz(self):
        return OrderedDict({'xyz': self.frz_xyz, 'f_dc': self.frz_features_dc, 'f_rest': self.frz_features_rest, 'sc': self.frz_scaling, 'rot': self.frz_rotation, 'op': self.frz_opacity})

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, ignore_colors: bool=False):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if ignore_colors:
            fused_color = RGB2SH(torch.tensor(0.5 * np.ones_like(pcd.points)).float().cuda())
        else:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        print('Number of points at initialisation : ', fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 1e-07)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 4)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device='cuda')
        rots[:, 0] = 1
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device='cuda'))
        self._latents = OrderedDict([(n, None) for n in self.param_names])
        init = self.latent_decoders['xyz'].invert(fused_point_cloud)
        self._latents['xyz'] = nn.Parameter(init.requires_grad_(True))
        init = features[:, :, 0:1].transpose(1, 2).contiguous()
        if self.latent_args.f_dc_invert_type == 'autoenc' and (not isinstance(self.latent_decoders['f_dc'], DecoderIdentity)):
            (init, decoder_state_dict) = init_latents(self.latent_args, fused_color.flatten(1).cuda(), 'f_dc', lambda_distortion=0.0)
            self.latent_decoders['f_dc'].load_state_dict(decoder_state_dict)
        elif not isinstance(self.latent_decoders['f_dc'], DecoderIdentity):
            init = self.latent_decoders['f_dc'].invert(fused_color.flatten(start_dim=1).contiguous().cuda())
        self._latents['f_dc'] = nn.Parameter(init.requires_grad_(True))
        init = features[:, :, 1:].transpose(1, 2).contiguous()
        if isinstance(self.latent_decoders['f_rest'], LatentDecoder):
            init = torch.zeros((features.size(0), self.latent_decoders['f_rest'].latent_dim)).to(init).contiguous()
        self._latents['f_rest'] = nn.Parameter(init.requires_grad_(True))
        if self.latent_args.sc_invert_type == 'autoenc' and (not isinstance(self.latent_decoders['sc'], DecoderIdentity)):
            (init, decoder_state_dict) = init_latents(self.latent_args, scales, 'sc', lambda_distortion=0.0)
            self.latent_decoders['sc'].load_state_dict(decoder_state_dict)
        else:
            init = self.latent_decoders['sc'].invert(scales)
        self._latents['sc'] = nn.Parameter(init.requires_grad_(True))
        if self.latent_args.rot_invert_type == 'autoenc' and (not isinstance(self.latent_decoders['rot'], DecoderIdentity)):
            (init, decoder_state_dict) = init_latents(self.latent_args, rots, 'rot', lambda_distortion=0.0)
            self.latent_decoders['rot'].load_state_dict(decoder_state_dict)
        else:
            init = self.latent_decoders['rot'].invert(rots)
        self._latents['rot'] = nn.Parameter(init.requires_grad_(True))
        if self.latent_args.op_invert_type == 'autoenc' and (not isinstance(self.latent_decoders['op'], DecoderIdentity)):
            (init, decoder_state_dict) = init_latents(self.latent_args, opacities, 'op', lambda_distortion=0.0)
            self.latent_decoders['op'].load_state_dict(decoder_state_dict)
        else:
            init = self.latent_decoders['op'].invert(opacities)
        self._latents['op'] = nn.Parameter(init.requires_grad_(True))
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device='cuda')
        self.mask_xyz.data = torch.ones_like(self._opacity).bool()
        self.mask_features_dc.data = torch.ones_like(self._opacity).bool()
        self.mask_features_rest.data = torch.ones_like(self._opacity).bool()
        self.mask_scaling.data = torch.ones_like(self._opacity).bool()
        self.mask_rotation.data = torch.ones_like(self._opacity).bool()
        self.mask_opacity.data = torch.ones_like(self._opacity).bool()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.infl_accum = torch.zeros(self.get_xyz.shape[0], device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.infl_denom = torch.zeros(self.get_xyz.shape[0], device='cuda')
        self.added_mask = None
        self.lr_scaling = OrderedDict()
        self.gate_params = OrderedDict()
        for (i, param) in enumerate(self.param_names):
            decoder = self.latent_decoders[param]
            if type(decoder) == DecoderIdentity or (type(decoder) == LatentDecoderRes and decoder.identity):
                self.lr_scaling[param] = 1.0
            else:
                self.lr_scaling[param] = training_args.latents_lr_scaling[i]
            self.gate_params[param] = self.latent_args.gate_params[i] != 'none'
        self.orig_lr = OrderedDict({'xyz': training_args.position_lr_init, 'f_dc': training_args.features_dc_lr, 'f_rest': training_args.features_rest_lr, 'sc': training_args.scaling_lr, 'rot': training_args.rotation_lr, 'op': training_args.opacity_lr})
        lr = {'xyz': training_args.position_lr_init * self.spatial_lr_scale * self.lr_scaling['xyz'], 'f_dc': training_args.features_dc_lr * self.lr_scaling['f_dc'], 'f_rest': training_args.features_rest_lr * self.lr_scaling['f_rest'], 'sc': training_args.scaling_lr * self.lr_scaling['sc'], 'rot': training_args.rotation_lr * self.lr_scaling['rot'], 'op': training_args.opacity_lr * self.lr_scaling['op']}
        l = []
        for (i, param) in enumerate(self.param_names):
            l += [{'params': [self._latents[param]], 'lr': lr[param], 'name': param}]
            if not isinstance(self.latent_decoders[param], DecoderIdentity):
                l += [{'params': self.latent_decoders[param].parameters(), 'lr': training_args.ldecs_lr[i], 'name': f'ldec_{param}'}]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale, lr_final=training_args.position_lr_final * self.spatial_lr_scale, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)

    def update_residuals(self):
        """Initialize residual encoders for temporal compression in subsequent frames."""
        atts = self.get_atts
        self.xyz_before = self.get_xyz.clone()
        self.mapping = torch.arange(self.xyz_before.shape[0], device='cuda')
        for (i, att_name) in enumerate(atts):
            if self.latent_args.quant_type[i] == 'sq_res':
                decoded_att = self.latent_decoders[att_name](self._latents[att_name])
                if self.frame_idx == 2:
                    assert isinstance(self.latent_decoders[att_name], DecoderIdentity)
                    decoder = LatentDecoderRes(latent_dim=self.latent_args.latent_dim[i], feature_dim=self.feature_dims[att_name], ldecode_matrix=self.latent_args.ldecode_matrix[i], latent_norm=self.latent_args.latent_norm[i], num_layers_dec=self.latent_args.num_layers_dec[i], hidden_dim_dec=self.latent_args.hidden_dim_dec[i], activation=self.latent_args.activation[i], use_shift=self.latent_args.use_shift[i], ldec_std=self.latent_args.ldec_std[i], final_activation=self.latent_args.final_activation[i]).cuda()
                    self.latent_decoders[att_name] = decoder
                else:
                    decoder = self.latent_decoders[att_name]
                    assert isinstance(self.latent_decoders[att_name], LatentDecoderRes)
                self.latent_decoders[att_name].frame_idx = self.frame_idx
                if 'f_' in att_name and self.frame_idx == 2:
                    decoder.init_decoded(decoded_att.reshape(decoded_att.shape[0], -1))
                else:
                    decoder.init_decoded(decoded_att)
                if type(self.latent_decoders[att_name]) == LatentDecoderRes and self.latent_args.quant_after[i] > 0.0:
                    self.latent_decoders[att_name].identity = True
                    decoder = self.latent_decoders[att_name]
                    latent = torch.zeros_like(decoder.decoded_att)
                    self._latents[att_name] = nn.Parameter(latent.requires_grad_(True))
                else:
                    self._latents[att_name] = nn.Parameter(torch.zeros((self._latents[att_name].shape[0], self.latent_args.latent_dim[i]), dtype=torch.float, device='cuda').requires_grad_(True))

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, is_dynamic=False):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros(n_init_points, device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        if selected_pts_mask.sum().item() == 0:
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device='cuda')
        samples = torch.normal(mean=means, std=stds, generator=self.split_generator)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        residual = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        new_xyz = residual + self.get_xyz[selected_pts_mask].repeat(N, 1)
        if type(self.latent_decoders['xyz']) == LatentDecoderRes:
            decoded_att = self.latent_decoders['xyz'].decoded_att[selected_pts_mask].repeat(N, 1)
            new_xyz = self.latent_decoders['xyz'].invert(new_xyz - decoded_att)
        else:
            new_xyz = self.latent_decoders['xyz'].invert(new_xyz)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        if type(self.latent_decoders['sc']) == LatentDecoderRes:
            decoded_att = self.latent_decoders['sc'].decoded_att[selected_pts_mask].repeat(N, 1)
            new_scaling = self.latent_decoders['sc'].invert(new_scaling - decoded_att)
        else:
            new_scaling = self.latent_decoders['sc'].invert(new_scaling)
        new_rotation = self._latents['rot'][selected_pts_mask].repeat(N, 1)
        if isinstance(self.latent_decoders['f_dc'], DecoderIdentity) or (type(self.latent_decoders['f_dc']) == LatentDecoderRes and self.latent_decoders['f_dc'].identity):
            new_features_dc = self._latents['f_dc'][selected_pts_mask].repeat(N, 1, 1)
        else:
            new_features_dc = self._latents['f_dc'][selected_pts_mask].repeat(N, 1)
        if isinstance(self.latent_decoders['f_rest'], DecoderIdentity) or (type(self.latent_decoders['f_rest']) == LatentDecoderRes and self.latent_decoders['f_rest'].identity):
            new_features_rest = self._latents['f_rest'][selected_pts_mask].repeat(N, 1, 1)
        else:
            new_features_rest = self._latents['f_rest'][selected_pts_mask].repeat(N, 1)
        new_opacity = self._latents['op'][selected_pts_mask].repeat(N, 1)
        new_flow = None
        new_motion_label = None
        new_offset_3d = None
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_flow, new_motion_label, new_offset_3d, split_mask=selected_pts_mask, N=N)
        self.added_mask = torch.cat((self.added_mask, torch.ones(new_xyz.shape[0]).bool().to(self._latents['xyz'].device)))
        if is_dynamic:
            new_mapping = torch.cat((self.mapping, torch.nonzero(selected_pts_mask).repeat(N, 1).flatten()), dim=0)
            if new_mapping.max() >= self.xyz_before.shape[0]:
                new_mapping[new_mapping >= self.xyz_before.shape[0]] = new_mapping[new_mapping[new_mapping >= self.xyz_before.shape[0]]]
            self.mapping = new_mapping
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self.prune_points(prune_filter)
        if is_dynamic:
            new_mapping = self.mapping[~prune_filter]
            if new_mapping.max() >= self.xyz_before.shape[0]:
                new_mapping[new_mapping >= self.xyz_before.shape[0]] = new_mapping[new_mapping[new_mapping >= self.xyz_before.shape[0]]]
            self.mapping = new_mapping

    def densify_and_clone(self, grads, grad_threshold, scene_extent, is_dynamic=False):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        if self.added_mask is None:
            self.added_mask = torch.zeros(self._latents['xyz'].shape[0]).bool().to(self._latents['xyz'].device)
        if selected_pts_mask.sum().item() == 0:
            return
        new_xyz = self._latents['xyz'][selected_pts_mask]
        new_features_dc = self._latents['f_dc'][selected_pts_mask]
        new_features_rest = self._latents['f_rest'][selected_pts_mask]
        new_scaling = self._latents['sc'][selected_pts_mask]
        new_rotation = self._latents['rot'][selected_pts_mask]
        new_opacities = self._latents['op'][selected_pts_mask]
        new_flow = None
        new_motion_labels = None
        new_offset_3d = None
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_flow, new_motion_labels, new_offset_3d, clone_mask=selected_pts_mask)
        self.added_mask = torch.cat((self.added_mask, torch.ones(new_xyz.shape[0]).bool().to(self._latents['xyz'].device)))
        if is_dynamic:
            new_mapping = torch.cat((self.mapping, torch.nonzero(selected_pts_mask).flatten()), dim=0)
            if new_mapping.max() >= self.xyz_before.shape[0]:
                new_mapping[new_mapping >= self.xyz_before.shape[0]] = new_mapping[new_mapping[new_mapping >= self.xyz_before.shape[0]]]
            self.mapping = new_mapping

    def _xyz_fixed(self):
        return self.latent_decoders['xyz'](self._latents['xyz'])

    @property
    def get_motion_label(self):
        raise RuntimeError('MORGS motion confidence is disabled')

    @property
    def get_flow(self):
        raise RuntimeError('MORGS flow module is disabled')

    @property
    def get_resi(self):
        raise RuntimeError('MORGS flow rendering is disabled')

    @property
    def get_offset_3d(self):
        raise RuntimeError('MORGS motion offset is disabled')
