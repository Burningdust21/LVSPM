import torch
import torch.nn as nn
import torch.nn.functional as nnf

import addict


class SafeDict(addict.Dict):
    def __missing__(self, name):
        if object.__getattribute__(self, '__frozen'):
            raise AttributeError(name)
        return self.__class__(__parent=self, __key=name)

    def __setitem__(self, name, value):
        is_frozen = (hasattr(self, '__frozen') and
                     object.__getattribute__(self, '__frozen'))
        assert not is_frozen, f"Cannot set field '{name}': frozen dict."

        super().__setitem__(name, value)
        try:
            p = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')


class TransformInput(nn.Module):
    def __init__(self, wo_grad=True):
        super().__init__()
        self.wo_grad = wo_grad

    def forward(self, image, intrinsics, c2w, patch_size=None):
        # pose stage model requires grad
        grad_context = torch.no_grad if self.wo_grad else torch.enable_grad
        with grad_context():
            return self._forward(image, intrinsics, c2w, patch_size)
    def _forward(self, image, intrinsics, c2w, patch_size=None):
        """transform input image before feeding into transformer

        Args:
            data_batch: SafeDict
            patch_size: int, optional

        Returns:
            image: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
        """
        # image, fxfycxcy, c2w, index = (
        #     data_batch.image,
        #     data_batch.fxfycxcy,
        #     data_batch.c2w,
        #     data_batch.index,
        # )

        assert image.dim() == 5, f"image dim should be 5, but got {image.dim()}"
        # assert (
        #     fxfycxcy.dim() == 3
        # ), f"fxfycxcy dim should be 3, but got {fxfycxcy.dim()}"
        # assert c2w.dim() == 4, f"c2w dim should be 4, but got {c2w.dim()}"

        fxfycxcy = torch.stack(
            [intrinsics[..., 0,0], intrinsics[..., 1,1],
             intrinsics[..., 0,2], intrinsics[..., 1,2]], dim=-1
        )
        b, v, c, h, w = image.size()
        image = image.reshape(b * v, c, h * w)
        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        c2w = c2w.reshape(b * v, 4, 4)

        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(image.device), x.to(image.device)

        # [h, w]
        y_norm, x_norm = (y + 0.5) / h * 2 - 1, (x + 0.5) / w * 2 - 1
        # [b, v, 2, h, w]
        xy_norm = torch.stack([x_norm, y_norm], dim=0)[None, None, :, :, :].expand(
            b, v, -1, -1, -1
        )

        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d_cam = ray_d.clone()
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_d_cam = ray_d_cam / torch.norm(ray_d_cam, dim=2, keepdim=True)
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        # fetch color, ray_o, ray_d for all patch centers
        ray_color_patch, ray_o_patch, ray_d_patch = None, None, None
        ray_xy_norm_patch, proj_mat = None, None
        if patch_size is not None:
            start_patch_center = patch_size / 2.0
            y, x = torch.meshgrid(
                torch.arange(h // patch_size) * patch_size + start_patch_center,
                torch.arange(w // patch_size) * patch_size + start_patch_center,
                indexing="ij",
            )
            y, x = y.to(image.device), x.to(image.device)
            x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
            y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)

            ray_xy_norm_patch = torch.stack(
                [x / w, y / h], dim=2
            )  # use [0,1] for patch center
            K_norm = (
                torch.eye(3, device=image.device).unsqueeze(0).repeat(b * v, 1, 1)
            )  # [b*v, 3, 3]
            K_norm[:, 0, 0] = fxfycxcy[:, 0] / w
            K_norm[:, 1, 1] = fxfycxcy[:, 1] / h
            K_norm[:, 0, 2] = fxfycxcy[:, 2] / w
            K_norm[:, 1, 2] = fxfycxcy[:, 3] / h
            w2c = torch.inverse(c2w)  # [b*v, 4, 4]
            proj_mat = torch.bmm(K_norm, w2c[:, :3, :4])  # [b*v, 3, 4]
            proj_mat = proj_mat.reshape(b * v, 12)
            proj_mat = proj_mat / (proj_mat.norm(dim=1, keepdim=True) + 1e-6)
            proj_mat = proj_mat.reshape(b * v, 3, 4)
            proj_mat = proj_mat * proj_mat[:, 0:1, 0:1].sign()

            # fetch colors for all patch centers
            ray_color_patch = (
                nnf.grid_sample(
                    image.reshape(b * v, c, h, w),
                    torch.stack([x / w * 2.0 - 1.0, y / h * 2.0 - 1.0], dim=2).reshape(
                        b * v, -1, 1, 2
                    ),
                    align_corners=False,
                )
                .squeeze(-1)
                .permute(0, 2, 1)
            ).contiguous()  # [b*v, h'*w', c]

            x = (x - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
            y = (y - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
            z = torch.ones_like(x)
            ray_d_patch = torch.stack([x, y, z], dim=2)  # [b*v, h'*w', 3]
            ray_d_patch = torch.bmm(
                ray_d_patch, c2w[:, :3, :3].transpose(1, 2)
            )  # [b*v, h'*w', 3]
            ray_d_patch = ray_d_patch / torch.norm(
                ray_d_patch, dim=2, keepdim=True
            )  # [b*v, h'*w', 3]
            ray_o_patch = c2w[:, :3, 3][:, None, :].expand_as(
                ray_d_patch
            )  # [b*v, h'*w', 3]

            n_patch = ray_color_patch.size(1)
            ray_color_patch = ray_color_patch.reshape(b, v, n_patch, c)
            ray_o_patch = ray_o_patch.reshape(b, v, n_patch, 3)
            ray_d_patch = ray_d_patch.reshape(b, v, n_patch, 3)
            ray_xy_norm_patch = ray_xy_norm_patch.reshape(b, v, n_patch, 2)
            proj_mat = proj_mat.reshape(b, v, 3, 4)

        ray_o = ray_o.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
        ray_d = ray_d.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
        ray_d_cam = ray_d_cam.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
        image = image.reshape(b, v, c, h, w)
        fxfycxcy = fxfycxcy.reshape(b, v, 4)
        c2w = c2w.reshape(b, v, 4, 4)
        ret = SafeDict(
            image=image,
            ray_o=ray_o,
            ray_d=ray_d,
            ray_d_cam=ray_d_cam,
            fxfycxcy=fxfycxcy,
            c2w=c2w,
            # index=index,
            xy_norm=xy_norm,
            ray_color_patch=ray_color_patch,
            ray_o_patch=ray_o_patch,
            ray_d_patch=ray_d_patch,
            ray_xy_norm_patch=ray_xy_norm_patch,
            proj_mat=proj_mat,
        )
        return ret
