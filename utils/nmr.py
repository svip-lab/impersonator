import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_renderer as nr
import utils.mesh as mesh


class SMPLRenderer(nn.Module):
    def __init__(self, faces, uv_map_path, map_name='par', tex_size=3, image_size=224,
                 flength=500., anti_aliasing=True, fill_back=True, background_color=(0, 0, 0), has_front_map=None):
        """
        :param image_size:  224
        :param flength: 500
        :param anti_aliasing:
        :param fill_back:
        """
        super(SMPLRenderer, self).__init__()

        self.background_color = background_color
        self.anti_aliasing = anti_aliasing
        self.flength = flength
        self.image_size = image_size
        self.fill_back = fill_back
        self.map_name = map_name

        self.tex_size = tex_size
        self.base_nf = faces.shape[0]
        self.coords = self.create_coords(tex_size).cuda()

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, list(reversed(range(faces.shape[-1])))]), dim=0).detach()

        self.faces = faces.cuda()
        self.nf = faces.shape[0]

        # (nf, T*T, 2)
        self.img2uv_sampler = torch.FloatTensor(mesh.create_uvsampler(uv_map_path, tex_size=tex_size)).cuda()
        self.map_fn = torch.FloatTensor(mesh.create_mapping(map_name, uv_map_path, contain_bg=True, fill_back=fill_back)).cuda()

        if has_front_map:
            self.front_map_fn = torch.FloatTensor(mesh.create_mapping('par', uv_map_path, contain_bg=True, fill_back=fill_back)).cuda()
        else:
            self.front_map_fn = None

        # light
        self.light_intensity_ambient = 1
        self.light_intensity_directional = 0
        self.light_color_ambient = [1, 1, 1]
        self.light_color_directional = [1, 1, 1]
        self.light_direction = [0, 1, 0]

        self.rasterizer_eps = 1e-3

    def set_ambient_light(self, lia=0.5, lid=0.5, background=(-1, -1, -1)):
        self.light_intensity_ambient = lia
        self.light_intensity_directional = lid
        self.background_color = background

    def set_tex_size(self, tex_size):
        del self.coords
        self.coords = self.create_coords(tex_size).cuda()

    def weak_projection(self, cam, vertices):
        """
        :param cam: (N, 3), [s, tx, ty]
        :param vertices: (N, number_points, 3)
        :return:
        """
        p = self.image_size / 2.0
        tz = self.flength / (p * cam[:, 0:1])
        verts2d = self.batch_orth_proj_idrot(cam, vertices)
        trans = vertices[:, :, 2:3] + tz[:, None, :]
        vertices = torch.cat((verts2d, trans), dim=-1)
        vertices[:, :, 1] *= -1
        return vertices

    def forward(self, cam, vertices, uv_imgs, is_uv_sampler=True, reverse_yz=False, get_fim=False):
        bs = cam.shape[0]
        faces = self.faces.repeat(bs, 1, 1)

        if is_uv_sampler:
            samplers = self.img2uv_sampler.repeat(bs, 1, 1, 1)
        else:
            samplers = self.dynamic_sampler(cam, vertices, faces)

        textures = self.extract_tex(uv_imgs, samplers)

        images, fim = self.render(cam, vertices, textures, faces, reverse_yz=reverse_yz, get_fim=get_fim)

        if get_fim:
            return images, textures, fim
        else:
            return images, textures

    def render(self, cam, vertices, textures, faces=None, near=0.1, far=25.0, reverse_yz=False, get_fim=False):
        # if reverse_yz:
        #     vertices, cam = vertices.clone(), cam.clone()
        #     vertices[:, :, 1] = -vertices[:, :, 1]  # y取反
        #     cam[:, 2] = -cam[:, 2]  # cam y取反

        if faces is None:
            bs = cam.shape[0]
            faces = self.faces.repeat(bs, 1, 1)

        # fill back
        # if self.fill_back:
            # faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            # textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

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

        vertices = self.weak_projection(cam, vertices)
        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, near, far, self.rasterizer_eps,
            self.background_color)

        fim = None
        if get_fim:
            fim = nr.rasterize_face_index_map(faces, self.image_size, False)

        return images, fim

    def render_depth(self, cam, vertices):
        bs = cam.shape[0]
        faces = self.faces.repeat(bs, 1, 1)
        # if self.fill_back:
        #     faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        vertices = self.weak_projection(cam, vertices)

        # rasterization
        faces = self.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render_silhouettes(self, cam, vertices, reverse_y=False):
        bs = cam.shape[0]
        faces = self.faces.repeat(bs, 1, 1)
        if reverse_y:
            vertices, cam = vertices.clone(), cam.clone()
            vertices[:, :, 1] = -vertices[:, :, 1]  # y取反
            cam[:, 2] = -cam[:, 2]  # cam y取反

        # if self.fill_back:
        #     faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        vertices = self.weak_projection(cam, vertices)

        # rasterization
        faces = self.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def infer_face_index_map(self, cam, vertices):
        bs = cam.shape[0]
        faces = self.faces.repeat(bs, 1, 1)

        # if self.fill_back:
        #     faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        vertices = self.weak_projection(cam, vertices)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        fim = nr.rasterize_face_index_map(faces, self.image_size, False)

        return fim

    def encode_fim(self, cam, vertices, fim=None, transpose=True):

        if fim is None:
            fim = self.infer_face_index_map(cam, vertices)

        fim_enc = self.map_fn[fim.long()]

        if transpose:
            fim_enc = fim_enc.permute(0, 3, 1, 2)

        return fim_enc, fim

    def encode_front_fim(self, fim, transpose=True):

        fim_enc = self.front_map_fn[fim.long()]

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

    def debug_textures(self):
        return torch.ones((self.nf, self.tex_size, self.tex_size, self.tex_size, 3), dtype=torch.float32)


class SMPLRendererTrainer(SMPLRenderer):
    def __init__(self, faces, uv_map_path, map_name='par', tex_size=3, image_size=224,
                 flength=500., anti_aliasing=True, fill_back=True, background_color=(0, 0, 0), has_front_map=None):
        """
        :param image_size:  224
        :param flength: 500
        :param anti_aliasing:
        :param fill_back:
        """
        nn.Module.__init__(self)
        self.background_color = background_color
        self.anti_aliasing = anti_aliasing
        self.flength = flength
        self.image_size = image_size
        self.fill_back = fill_back
        self.map_name = map_name

        self.tex_size = tex_size
        self.base_nf = faces.shape[0]
        # self.coords = self.create_coords(tex_size).cuda()
        self.register_buffer('coords', self.create_coords(tex_size))

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, list(reversed(range(faces.shape[-1])))]), dim=0).detach()

        # self.faces = faces.cuda()
        self.register_buffer('faces', faces)
        self.nf = faces.shape[0]

        # (nf, T*T, 2)
        img2uv_sampler = torch.FloatTensor(mesh.create_uvsampler(uv_map_path, tex_size=tex_size))
        map_fn = torch.FloatTensor(mesh.create_mapping(map_name, uv_map_path, contain_bg=True, fill_back=fill_back))

        self.register_buffer('img2uv_sampler', img2uv_sampler)
        self.register_buffer('map_fn', map_fn)

        if has_front_map:
            front_map_fn = torch.FloatTensor(mesh.create_mapping('par', uv_map_path, contain_bg=True, fill_back=fill_back))
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

