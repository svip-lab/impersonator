import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import neural_renderer as nr
from . import mesh


def orthographic_proj_withz_idrot(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 3: [sc, tx, ty]
    No rotation!
    Orth preserving the z.
    sc * ( x + [tx; ty])
    as in HMR..
    """
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    # proj = scale * X
    proj = X

    proj_xy = scale * (proj[:, :, :2] + trans)
    proj_z = proj[:, :, 2, None] + offset_z

    return torch.cat((proj_xy, proj_z), 2)


def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    sc * ( x + [tx; ty])
    as in HMR..
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    # proj = scale * X_rot
    proj = X_rot

    proj_xy = scale * (proj[:, :, :2] + trans)
    proj_z = proj[:, :, 2, None] + offset_z

    return torch.cat((proj_xy, proj_z), 2)


def quat_rotate(X, q):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        q: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]] * 0 + 1
    q = torch.unsqueeze(q, 1) * ones_x

    q_conj = torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)
    X = torch.cat([X[:, :, [0]] * 0, X], dim=-1)

    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]


def hamilton_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
    q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
    q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
    q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


class SMPLRenderer(nn.Module):
    def __init__(self, face_path='assets/pretrains/smpl_faces.npy',
                 uv_map_path='assets/pretrains/mapper.txt', map_name='uv_seg', tex_size=3, image_size=256,
                 anti_aliasing=True, fill_back=False, background_color=(0, 0, 0), viewing_angle=30, near=0.1, far=25.0,
                 has_front=False):
        """
        Args:
            face_path:
            uv_map_path:
            map_name:
            tex_size:
            image_size:
            anti_aliasing:
            fill_back:
            background_color:
            viewing_angle:
            near:
            far:
            has_front:
        """

        super(SMPLRenderer, self).__init__()

        self.background_color = background_color
        self.anti_aliasing = anti_aliasing
        self.image_size = image_size
        self.fill_back = fill_back
        self.map_name = map_name

        faces = np.load(face_path)
        self.tex_size = tex_size
        self.base_nf = faces.shape[0]
        self.register_buffer('coords', self.create_coords(tex_size))

        # fill back
        if self.fill_back:
            faces = np.concatenate((faces, faces[:, ::-1]), axis=0)

        faces = torch.tensor(faces.astype(np.int32)).int()
        self.nf = faces.shape[0]
        self.register_buffer('faces', faces)

        # (nf, T*T, 2)
        img2uv_sampler = torch.tensor(mesh.create_uvsampler(uv_map_path, tex_size=tex_size)).float()
        map_fn = torch.tensor(mesh.create_mapping(map_name, uv_map_path, contain_bg=True,
                                                  fill_back=fill_back)).float()
        self.register_buffer('img2uv_sampler', img2uv_sampler)
        self.register_buffer('map_fn', map_fn)

        back_map_fn = torch.tensor(mesh.create_mapping('back', uv_map_path, contain_bg=True,
                                                       fill_back=fill_back)).float()
        self.register_buffer('back_map_fn', back_map_fn)

        if has_front:
            front_map_fn = torch.tensor(mesh.create_mapping('front', uv_map_path, contain_bg=True,
                                                            fill_back=fill_back)).float()
            self.register_buffer('front_map_fn', front_map_fn)
        else:
            self.front_map_fn = None

        # light
        self.light_intensity_ambient = 1
        self.light_intensity_directional = 0
        self.light_color_ambient = [1, 1, 1]
        self.light_color_directional = [1, 1, 1]
        self.light_direction = [0, 1, 0]

        self.rasterizer_eps = 1e-3

        # project function and camera
        self.near = near
        self.far = far
        self.proj_func = orthographic_proj_withz_idrot
        self.viewing_angle = viewing_angle
        self.eye = [0, 0, -(1. / np.tan(np.radians(self.viewing_angle)) + 1)]

    def set_ambient_light(self, int_dir=0.3, int_amb=0.7, direction=(1, 0.5, 1)):
        self.light_intensity_directional = int_dir
        self.light_intensity_ambient = int_amb
        if direction is not None:
            self.light_direction = direction

    def set_bgcolor(self, color=(-1, -1, -1)):
        self.background_color = color

    def set_tex_size(self, tex_size):
        del self.coords
        self.coords = self.create_coords(tex_size).cuda()

    def forward(self, cam, vertices, uv_imgs, dynamic=True, get_fim=False):
        bs = cam.shape[0]
        faces = self.faces.repeat(bs, 1, 1)

        if dynamic:
            samplers = self.dynamic_sampler(cam, vertices, faces)
        else:
            samplers = self.img2uv_sampler.repeat(bs, 1, 1, 1)

        textures = self.extract_tex(uv_imgs, samplers)

        images, fim = self.render(cam, vertices, textures, faces, get_fim=get_fim)

        if get_fim:
            return images, textures, fim
        else:
            return images, textures

    def render(self, cam, vertices, textures, faces=None, get_fim=False):
        if faces is None:
            bs = cam.shape[0]
            faces = self.faces.repeat(bs, 1, 1)

        # lighting is inplace operation
        textures = textures.clone()
        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(faces, textures, self.image_size, self.anti_aliasing,
                              self.near, self.far, self.rasterizer_eps, self.background_color)
        fim = None
        if get_fim:
            fim = nr.rasterize_face_index_map(faces, image_size=self.image_size, anti_aliasing=False,
                                              near=self.near, far=self.far, eps=self.rasterizer_eps)

        return images, fim

    def render_fim(self, cam, vertices, faces=None):
        if faces is None:
            bs = cam.shape[0]
            faces = self.faces.repeat(bs, 1, 1)

        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        fim = nr.rasterize_face_index_map(faces, self.image_size, False)
        return fim

    def render_fim_wim(self, cam, vertices, faces=None):
        if faces is None:
            bs = cam.shape[0]
            faces = self.faces.repeat(bs, 1, 1)

        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        fim, wim = nr.rasterize_face_index_map_and_weight_map(faces, self.image_size, False)
        return faces, fim, wim

    def render_depth(self, cam, vertices):
        raise NotImplementedError

        # bs = cam.shape[0]
        # faces = self.faces.repeat(bs, 1, 1)
        # # if self.fill_back:
        # #     faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
        #
        # vertices = self.weak_projection(cam, vertices)
        #
        # # rasterization
        # faces = self.vertices_to_faces(vertices, faces)
        # images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        # return images

    def render_silhouettes(self, cam, vertices, faces=None):
        if faces is None:
            bs = cam.shape[0]
            faces = self.faces.repeat(bs, 1, 1)

        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        proj_verts[:, :, 1] *= -1
        # calculate the look_at vertices.
        vertices = nr.look_at(proj_verts, self.eye)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def infer_face_index_map(self, cam, vertices):
        raise NotImplementedError
        # bs = cam.shape[0]
        # faces = self.faces.repeat(bs, 1, 1)
        #
        # # if self.fill_back:
        # #     faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
        #
        # vertices = self.weak_projection(cam, vertices)
        #
        # # rasterization
        # faces = nr.vertices_to_faces(vertices, faces)
        # fim = nr.rasterize_face_index_map(faces, self.image_size, False)

        # return fim

    def encode_fim(self, cam, vertices, fim=None, transpose=True, map_fn=None):

        if fim is None:
            fim = self.infer_face_index_map(cam, vertices)

        if map_fn is not None:
            fim_enc = map_fn[fim.long()]
        else:
            fim_enc = self.map_fn[fim.long()]

        if transpose:
            fim_enc = fim_enc.permute(0, 3, 1, 2)

        return fim_enc, fim

    def encode_front_fim(self, fim, transpose=True, front_fn=True):
        if front_fn:
            fim_enc = self.front_map_fn[fim.long()]
        else:
            fim_enc = self.back_map_fn[fim.long()]

        if transpose:
            fim_enc = fim_enc.permute(0, 3, 1, 2)

        return fim_enc

    def extract_tex_from_image(self, images, cam, vertices):
        bs = images.shape[0]
        faces = self.faces.repeat(bs, 1, 1)

        sampler = self.dynamic_sampler(cam, vertices, faces)  # (bs, nf, T*T, 2)

        tex = self.extract_tex(images, sampler)

        return tex

    def extract_tex(self, uv_img, uv_sampler):
        """
        :param uv_img: (bs, 3, h, w)
        :param uv_sampler: (bs, nf, T*T, 2)
        :return:
        """

        # (bs, 3, nf, T*T)
        tex = F.grid_sample(uv_img, uv_sampler)
        # (bs, 3, nf, T, T)
        tex = tex.view(-1, 3, self.nf, self.tex_size, self.tex_size)
        # (bs, nf, T, T, 3)
        tex = tex.permute(0, 2, 3, 4, 1)
        # (bs, nf, T, T, T, 3)
        tex = tex.unsqueeze(4).repeat(1, 1, 1, 1, self.tex_size, 1)

        return tex

    def dynamic_sampler(self, cam, vertices, faces):
        # ipdb.set_trace()
        points = self.batch_orth_proj_idrot(cam, vertices)  # (bs, nf, 2)
        faces_points = self.points_to_faces(points, faces)   # (bs, nf, 3, 2)
        # print(faces_points.shape)
        sampler = self.points_to_sampler(self.coords, faces_points)  # (bs, nf, T*T, 2)
        return sampler

    def project_to_image(self, cam, vertices):
        # set offset_z for persp proj
        proj_verts = self.proj_func(vertices, cam)
        # flipping the y-axis here to make it align with the image coordinate system!
        # proj_verts[:, :, 1] *= -1
        proj_verts = proj_verts[:, :, 0:2]
        return proj_verts

    def points_to_faces(self, points, faces=None):
        """
        :param points:
        :param faces
        :return:
        """
        bs, nv = points.shape[:2]
        device = points.device

        if faces is None:
            faces = self.faces.repeat(bs, 1, 1)
            # if self.fill_back:
            #     faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        points = points.reshape((bs * nv, 2))
        # pytorch only supports long and byte tensors for indexing
        return points[faces.long()]

    @staticmethod
    def compute_barycenter(f2vts):
        """

        :param f2vts:  N x F x 3 x 2
        :return: N x F x 2
        """

        # Compute alpha, beta (this is the same order as NMR)
        v2 = f2vts[:, :, 2]  # (nf, 2)
        v0v2 = f2vts[:, :, 0] - f2vts[:, :, 2]  # (nf, 2)
        v1v2 = f2vts[:, :, 1] - f2vts[:, :, 2]  # (nf, 2)

        fbc = v2 + 0.5 * v0v2 + 0.5 * v1v2

        return fbc

    @staticmethod
    def batch_orth_proj_idrot(camera, X):
        """
        X is N x num_points x 3
        camera is N x 3
        same as applying orth_proj_idrot to each N
        """

        # TODO check X dim size.
        # X_trans is (N, num_points, 2)
        X_trans = X[:, :, :2] + camera[:, None, 1:]
        # reshape X_trans, (N, num_points * 2)
        # --- * operation, (N, 1) x (N, num_points * 2) -> (N, num_points * 2)
        # ------- reshape, (N, num_points, 2)

        return camera[:, None, 0:1] * X_trans

    @staticmethod
    def points_to_sampler(coords, faces):
        """
        :param coords: [2, T*T]
        :param faces: [batch size, number of vertices, 3, 2]
        :return: [batch_size, number of vertices, T*T, 2]
        """

        # Compute alpha, beta (this is the same order as NMR)
        nf = faces.shape[1]
        v2 = faces[:, :, 2]  # (bs, nf, 2)
        v0v2 = faces[:, :, 0] - faces[:, :, 2]  # (bs, nf, 2)
        v1v2 = faces[:, :, 1] - faces[:, :, 2]  # (bs, nf, 2)

        # bs x  F x 2 x T*2
        samples = torch.matmul(torch.stack((v0v2, v1v2), dim=-1), coords) + v2.view(-1, nf, 2, 1)
        # bs x F x T*2 x 2 points on the sphere
        samples = samples.permute(0, 1, 3, 2)
        samples = torch.clamp(samples, min=-1.0, max=1.0)
        return samples

    @staticmethod
    def create_coords(tex_size=3):
        """
        :param tex_size: int
        :return: 2 x (tex_size * tex_size)
        """
        if tex_size == 1:
            step = 1
        else:
            step = 1 / (tex_size - 1)

        alpha_beta = torch.arange(0, 1+step, step, dtype=torch.float32).cuda()
        xv, yv = torch.meshgrid([alpha_beta, alpha_beta])

        coords = torch.stack([xv.flatten(), yv.flatten()], dim=0)

        return coords

    @staticmethod
    def create_meshgrid(image_size):
        """
        Args:
            image_size:

        Returns:
            (image_size, image_size, 2)
        """
        factor = torch.arange(0, image_size, dtype=torch.float32) / (image_size - 1)   # [0, 1]
        factor = (factor - 0.5) * 2
        xv, yv = torch.meshgrid([factor, factor])
        # grid = torch.stack([xv, yv], dim=-1)
        grid = torch.stack([yv, xv], dim=-1)
        return grid

    @staticmethod
    def get_vis_f2pts(f2pts, fims):
        """
        Args:
            f2pts: (bs, f, 3, 2) or (bs, f, 3, 3)
            fims:  (bs, 256, 256)

        Returns:

        """

        def get_vis(orig_f2pts, fim):
            """
            Args:
                orig_f2pts: (f, 3, 2) or (f, 3, 3)
                fim: (256, 256)

            Returns:
                vis_f2pts: (f, 3, 2)
            """
            vis_f2pts = torch.zeros_like(orig_f2pts) - 2.0
            # 0 is -1
            face_ids = fim.unique()[1:].long()
            vis_f2pts[face_ids] = orig_f2pts[face_ids]

            return vis_f2pts

        # import ipdb
        # ipdb.set_trace()
        if f2pts.dim() == 4:
            all_vis_f2pts = []
            bs = f2pts.shape[0]
            for i in range(bs):
                all_vis_f2pts.append(get_vis(f2pts[i], fims[i]))

            all_vis_f2pts = torch.stack(all_vis_f2pts, dim=0)

        else:
            all_vis_f2pts = get_vis(f2pts, fims)

        return all_vis_f2pts

    @staticmethod
    def set_null_f2pts(f2pts, fims):
        """
        Args:
            f2pts: (bs, f, 3, 2) or (bs, f, 3, 3)
            fims:  (bs, 256, 256)

        Returns:

        """

        def get_vis(orig_f2pts, fim):
            """
            Args:
                orig_f2pts: (f, 3, 2) or (f, 3, 3)
                fim: (256, 256)

            Returns:
                vis_f2pts: (f, 3, 2)
            """
            # 0 is -1
            face_ids = fim.unique()[1:].long()
            orig_f2pts[face_ids] = -2.0

            return orig_f2pts

        if f2pts.dim() == 4:
            all_vis_f2pts = []
            bs = f2pts.shape[0]
            for i in range(bs):
                all_vis_f2pts.append(get_vis(f2pts[i], fims[i]))

            all_vis_f2pts = torch.stack(all_vis_f2pts, dim=0)

        else:
            all_vis_f2pts = get_vis(f2pts, fims)

        return all_vis_f2pts

    def cal_transform(self, bc_f2pts, src_fim, dst_fim):
        """
        Args:
            bc_f2pts:
            src_fim:
            dst_fim:

        Returns:

        """
        device = bc_f2pts.device
        bs = src_fim.shape[0]
        # T = renderer.init_T.repeat(bs, 1, 1, 1)    # (bs, image_size, image_size, 2)
        T = (torch.zeros(bs, self.image_size, self.image_size, 2, device=device) - 2)
        # 2. calculate occlusion flows, (bs, no, 2)
        dst_ids = dst_fim != -1

        # 3. calculate tgt flows, (bs, nt, 2)

        for i in range(bs):
            Ti = T[i]

            tgt_i = dst_ids[i]

            # (nf, 2)
            tgt_flows = bc_f2pts[i, dst_fim[i, tgt_i].long()]  # (nt, 2)
            Ti[tgt_i] = tgt_flows

        return T

    def cal_bc_transform(self, src_f2pts, dst_fims, dst_wims):
        """
        Args:
            src_f2pts: (bs, 13776, 3, 2)
            dst_fims:  (bs, 256, 256)
            dst_wims:  (bs, 256, 256, 3)
        Returns:

        """
        bs = src_f2pts.shape[0]
        T = -2 * torch.ones((bs, self.image_size * self.image_size, 2), dtype=torch.float32, device=src_f2pts.device)

        for i in range(bs):
            # (13776, 3, 2)
            from_faces_verts_on_img = src_f2pts[i]

            # to_face_index_map
            to_face_index_map = dst_fims[i]

            # to_weight_map
            to_weight_map = dst_wims[i]

            # (256, 256) -> (256*256, )
            to_face_index_map = to_face_index_map.long().reshape(-1)
            # (256, 256, 3) -> (256*256, 3)
            to_weight_map = to_weight_map.reshape(-1, 3)

            to_exist_mask = (to_face_index_map != -1)
            # (exist_face_num,)
            to_exist_face_idx = to_face_index_map[to_exist_mask]
            # (exist_face_num, 3)
            to_exist_face_weights = to_weight_map[to_exist_mask]

            # (exist_face_num, 3, 2) * (exist_face_num, 3) -> sum -> (exist_face_num, 2)
            exist_smpl_T = (from_faces_verts_on_img[to_exist_face_idx] * to_exist_face_weights[:, :, None]).sum(dim=1)
            # (256, 256, 2)
            T[i, to_exist_mask] = exist_smpl_T

        T = T.view(bs, self.image_size, self.image_size, 2)

        # T = torch.clamp(-2, 2)

        return T

    def debug_textures(self):
        return torch.ones((self.nf, self.tex_size, self.tex_size, self.tex_size, 3), dtype=torch.float32)
