import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


def save_to_obj(verts, faces, path):
    """
    Save the SMPL model into .obj file.

    Parameter:
    ---------
    path: Path to save.

    """

    with open(path, 'w') as fp:
        fp.write('g\n')
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        fp.write('s off\n')


def load_obj(obj_file):
    with open(obj_file, 'r') as fp:
        verts = []
        faces = []
        vts = []
        vns = []
        faces_vts = []
        faces_vns = []

        for line in fp:
            line = line.rstrip()
            line_splits = line.split()
            prefix = line_splits[0]

            if prefix == 'v':
                verts.append(np.array([line_splits[1], line_splits[2], line_splits[3]], dtype=np.float32))

            elif prefix == 'vn':
                vns.append(np.array([line_splits[1], line_splits[2], line_splits[3]], dtype=np.float32))

            elif prefix == 'vt':
                vts.append(np.array([line_splits[1], line_splits[2]], dtype=np.float32))

            elif prefix == 'f':
                f = []
                f_vt = []
                f_vn = []
                for p_str in line_splits[1:4]:
                    p_split = p_str.split('/')
                    f.append(p_split[0])
                    f_vt.append(p_split[1])
                    f_vn.append(p_split[2])

                faces.append(np.array(f, dtype=np.int32) - 1)
                faces_vts.append(np.array(f_vt, dtype=np.int32) - 1)
                faces_vns.append(np.array(f_vn, dtype=np.int32) - 1)

            else:
                raise ValueError(prefix)

        obj_dict = {
            'vertices': np.array(verts, dtype=np.float32),
            'faces': np.array(faces, dtype=np.int32),
            'vts': np.array(vts, dtype=np.float32),
            'vns': np.array(vns, dtype=np.float32),
            'faces_vts': np.array(faces_vts, dtype=np.int32),
            'faces_vns': np.array(faces_vns, dtype=np.int32)
        }

        return obj_dict


def sample_textures(texture_flow, images):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N

    output: B x F x T x T x 3
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 3 x F x T*T
    samples = torch.nn.functional.grid_sample(images, flow_grid)
    # B x 3 x F x T x T
    samples = samples.view(-1, 3, F, T, T)
    # B x F x T x T x 3
    return samples.permute(0, 2, 3, 4, 1)


def get_spherical_coords(X):
    # X is N x 3
    rad = np.linalg.norm(X, axis=1)
    # Inclination
    theta = np.arccos(X[:, 2] / rad)
    # Azimuth
    phi = np.arctan2(X[:, 1], X[:, 0])

    # Normalize both to be between [-1, 1]
    vv = (theta / np.pi) * 2 - 1
    uu = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv], 1)


def compute_coords(tex_size):
    """
    :param tex_size:
    :return: (2, T*T)
    """
    alpha = np.arange(tex_size, dtype=np.float) / (tex_size - 1)
    beta = np.arange(tex_size, dtype=np.float) / (tex_size - 1)
    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])  # T*T x 2
    coords = torch.FloatTensor(coords.T)  # (2, T*T)
    return coords


def compute_uvsampler(verts, faces, tex_size=2):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    alpha = np.arange(tex_size, dtype=np.float) / (tex_size - 1)
    beta = np.arange(tex_size, dtype=np.float) / (tex_size - 1)
    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])  # 36 x 2
    vs = verts[faces]
    # Compute alpha, beta (this is the same order as NMR)
    v2 = vs[:, 2]  # (656, 3)
    v0v2 = vs[:, 0] - vs[:, 2]  # (656, 3)
    v1v2 = vs[:, 1] - vs[:, 2]  # (656, 3)
    # F x 3 x T*2
    samples = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 3, 1)
    # F x T*2 x 3 points on the sphere
    samples = np.transpose(samples, (0, 2, 1))

    # Now convert these to uv.
    uv = get_spherical_coords(samples.reshape(-1, 3))
    # uv = uv.reshape(-1, len(coords), 2)

    uv = uv.reshape(-1, tex_size, tex_size, 2)
    return uv


def compute_barycenter(f2vts):
    """

    :param f2vts:  F x 3 x 2
    :return: F x 2
    """

    # Compute alpha, beta (this is the same order as NMR)
    v2 = f2vts[:, 2]  # (nf, 2)
    v0v2 = f2vts[:, 0] - f2vts[:, 2]  # (nf, 2)
    v1v2 = f2vts[:, 1] - f2vts[:, 2]  # (nf, 2)

    fbc = v2 + 0.5 * v0v2 + 0.5 * v1v2

    return fbc


def get_f2vts(uv_mapping_path, fill_back=False):
    """
    For this mesh, pre-computes the bary-center coords.
    Returns F x 2
    """
    obj_info = load_obj(uv_mapping_path)

    vts = obj_info['vts']
    vts[:, 1] = 1 - vts[:, 1]
    # vts = vts * 2 - 1

    # F x (2 + 1) = F x 3
    vts = np.concatenate([vts, np.zeros((vts.shape[0], 1), dtype=np.float32)], axis=-1)
    faces = obj_info['faces_vts']

    if fill_back:
        faces = np.concatenate((faces, faces[:, ::-1]), axis=0)

    # F x 3 x 3
    f2vts = vts[faces]

    return f2vts


def create_uvf2vts(uv_mapping_path, add_z=True):
    obj_info = load_obj(uv_mapping_path)

    vts = obj_info['vts']
    vts[:, 1] = 1 - vts[:, 1]
    # vts = vts * 2 - 1

    # F x (2 + 1) = F x 3
    if add_z:
        vts = np.concatenate([vts, np.zeros((vts.shape[0], 1), dtype=np.float32) + 5], axis=-1)
    faces_vts = obj_info['faces_vts']

    # F x 3 x 3 or F x 3 x2
    uvf2vts = vts[faces_vts]
    return uvf2vts


def get_head_front_ids(nf, front_face_info, fill_back=False):

    if fill_back:
        half_nf = nf // 2

    with open(front_face_info, 'r') as reader:
        front_data = json.load(reader)

        faces = front_data['face']

        if fill_back:
            faces = faces + [f + half_nf for f in faces]

    return faces


def get_head_back_ids(nf, head_face_info, front_face_info, fill_back=False):

    if fill_back:
        half_nf = nf // 2

    with open(head_face_info, 'r') as reader:
        head_faces = set(json.load(reader)['face'])
        with open(front_face_info, 'r') as front_reader:
            front_faces = set(json.load(front_reader)['face'])

        faces = list(head_faces - front_faces)
        if fill_back:
            faces = faces + [f + half_nf for f in faces]

    return faces


def get_part_ids(nf, part_info, fill_back=False):
    if fill_back:
        half_nf = nf // 2
    with open(part_info, 'r') as reader:
        part_data = json.load(reader)

        part_names = sorted(part_data.keys())

        total_faces = set()
        ordered_faces = dict()
        for i, part_name in enumerate(part_names):
            part_vals = part_data[part_name]
            faces = part_vals['face']
            if fill_back:
                faces = faces + [f + half_nf for f in faces]
            ordered_faces[part_name] = faces
            total_faces |= set(faces)

        nf_counter = len(total_faces)
        assert nf_counter == nf, 'nf_counter = {}, nf = {}'.format(nf_counter, nf)

    return ordered_faces


def binary_mapping(nf):

    width = len(np.binary_repr(nf))
    map_fn = [np.array(list(map(int, np.binary_repr(i, width=width)))) for i in range(nf)]
    map_fn = np.stack(map_fn, axis=0)

    bg = np.zeros((1, width), dtype=np.float32) - 1.0

    return map_fn, bg


def ids_mapping(nf):
    map_fn = np.arange(0, 1, 1/nf, dtype=np.float32)
    bg = np.array([[-1]], dtype=np.float32)
    return map_fn, bg


def par_mapping(nf, part_info, fill_back=False):

    if fill_back:
        half_nf = nf // 2
    with open(part_info, 'r') as reader:
        part_data = json.load(reader)

        ndim = len(part_data) + 1 # 10
        map_fn = np.zeros((nf, ndim), dtype=np.float32)

        part_names = sorted(part_data.keys())

        total_faces = set()
        for i, part_name in enumerate(part_names):
            part_vals = part_data[part_name]
            faces = part_vals['face']

            if fill_back:
                faces = faces + [f + half_nf for f in faces]

            map_fn[faces, i] = 1.0
            total_faces |= set(faces)

        nf_counter = len(total_faces)
        assert nf_counter == nf, 'nf_counter = {}, nf = {}'.format(nf_counter, nf)

        # add bg
        bg = np.zeros((1, ndim), dtype=np.float32)
        bg[0, -1] = 1

        return map_fn, bg


def front_face_mapping(nf, front_face_info, fill_back=False):

    if fill_back:
        half_nf = nf // 2

    map_fn = np.zeros((nf, 1), dtype=np.float32)

    with open(front_face_info, 'r') as reader:
        front_data = json.load(reader)

        faces = front_data['face']

        if fill_back:
            faces = faces + [f + half_nf for f in faces]

        map_fn[faces] = 1.0

    # add bg
    bg = np.zeros((1, 1), dtype=np.float32)

    return map_fn, bg


def back_face_mapping(nf, head_face_info, front_face_info, fill_back=False):

    if fill_back:
        half_nf = nf // 2

    map_fn = np.zeros((nf, 1), dtype=np.float32)

    with open(head_face_info, 'r') as reader:
        head_faces = set(json.load(reader)['face'])
        with open(front_face_info, 'r') as front_reader:
            front_faces = set(json.load(front_reader)['face'])

        faces = list(head_faces - front_faces)
        if fill_back:
            faces = faces + [f + half_nf for f in faces]

        map_fn[faces] = 1.0

    # add bg
    bg = np.zeros((1, 1), dtype=np.float32)

    return map_fn, bg


def create_mapping(map_name, mapping_path='assets/pretrains/mapper.txt',
                   part_info='assets/pretrains/smpl_part_info.json',
                   front_info='assets/pretrains/front_facial.json',
                   head_info='assets/pretrains/head.json',
                   contain_bg=True, fill_back=False):
    """
    :param mapping_path:
    :param map_name:
            'uv'     -> (F + 1) x 2  (bg as -1)
            'ids'    -> (F + 1) x 1  (bg as -1)
            'binary' -> (F + 1) x 14 (bs as -1)
            'seg'    -> (F + 1) x 1  (bs as 0)
            'par'    -> (F + 1) x (10 + 1)
    :param part_info:
    :param front_info:
    :param contain_bg:
    :param fill_back:
    :return:
    """

    # F x C
    f2vts = get_f2vts(mapping_path, fill_back=fill_back)
    nf = f2vts.shape[0]

    if map_name == 'uv':
        fbc = compute_barycenter(f2vts)
        map_fn = fbc[:, 0:2]    # F x 2
        bg = np.array([[-1, -1]], dtype=np.float32)
    elif map_name == 'seg':
        map_fn = np.ones((nf, 1), dtype=np.float32)
        bg = np.array([[0]], dtype=np.float32)
    elif map_name == 'uv_seg':
        fbc = compute_barycenter(f2vts)
        map_fn = fbc    # F x 3
        bg = np.array([[0, 0, 1]], dtype=np.float32)
    elif map_name == 'par':
        map_fn, bg = par_mapping(nf, part_info)
    elif map_name == 'front':
        map_fn, bg = front_face_mapping(nf, front_info, fill_back=fill_back)
    elif map_name == 'head':
        map_fn, bg = front_face_mapping(nf, head_info, fill_back=fill_back)
    elif map_name == 'back':
        map_fn, bg = back_face_mapping(nf, head_info, front_info, fill_back=fill_back)
    elif map_name == 'ids':
        map_fn, bg = ids_mapping(nf)
    elif map_name == 'binary':
        map_fn, bg = binary_mapping(nf)
    else:
        raise ValueError('map name error {}'.format(map_name))

    if contain_bg:
        map_fn = np.concatenate([map_fn, bg], axis=0)

    return map_fn


def get_part_face_ids(part_type, mapping_path='assets/pretrains/mapper.txt',
                      part_info='assets/pretrains/smpl_part_info.json',
                      front_info='assets/pretrains/front_face_1.json',
                      head_info='assets/pretrains/head.json',
                      fill_back=False):
    # F x C
    f2vts = get_f2vts(mapping_path, fill_back=fill_back)
    nf = f2vts.shape[0]
    if part_type == 'head_front':
        faces = get_head_front_ids(nf, front_info, fill_back=fill_back)
    elif part_type == 'head_back':
        faces = get_head_back_ids(nf, head_info, front_info, fill_back=fill_back)
    elif part_type == 'head':
        raise NotImplementedError
    elif part_type == 'par':
        faces = get_part_ids(nf, part_info, fill_back=fill_back)
    else:
        raise ValueError('map name error {}'.format(part_type))

    return faces


def get_map_fn_dim(map_name):
    """
    :param map_name:
        'seg'    -> (F + 1) x 1  (bs as -1 or 0)
        'uv'     -> (F + 1) x 2  (bg as -1)
        'uv_seg' -> (F + 1) x 3  (bg as -1)
        'ids'    -> (F + 1) x 1  (bg as -1)
        'binary' -> (F + 1) x 15 (bs as -1)
        'par'    -> (F + 1) x (10 + 1)
    :return:
    """
    # F x C
    if map_name == 'seg':
        dim = 1
    elif map_name == 'uv':
        dim = 2
    elif map_name == 'uv_seg':
        dim = 3
    elif map_name == 'par':
        dim = 11
    elif map_name == 'ids':
        dim = 1
    elif map_name == 'binary':
        dim = 15
    else:
        raise ValueError('map name error {}'.format(map_name))

    return dim


def cvt_fim_enc(fim_enc, map_name):

    h, w, c = fim_enc.shape

    if map_name == 'uv':
        # (H, W, 2), bg is -1, -> (H, W, 3)
        img = np.ones((h, w, 3), dtype=np.float32)
        # print(fim_enc.shape)
        img[:, :, 0:2] = fim_enc[:, :, 0:2]
        img = np.transpose(img, axes=(2, 0, 1))

    elif map_name == 'seg':
        # (H, W, 1), bg is -1  -> (H, W)
        img = fim_enc[:, :, 0]

    elif map_name == 'uv_seg':
        # (H, W, 3) -> (H, W, 3)
        img = fim_enc.copy()
        img = np.transpose(img, axes=(2, 0, 1))

    elif map_name == 'par':
        # (H, W, C) -> (H, W)
        img = fim_enc.argmax(axis=-1)
        img = img.astype(np.float32)
        img /= img.max()

    elif map_name == 'ids':
        # (H, W, 1), bg is -1  -> (H, W)
        img = fim_enc[:, :, 0]

    elif map_name == 'binary':
        img = np.zeros((h, w), dtype=np.float32)

        def bin2int(bits):
            total = 0
            for shift, j in enumerate(bits[::-1]):
                if j:
                    total += 1 << shift
            return total

        for i in range(h):
            for j in range(w):
                val = bin2int(fim_enc[i, j, :])
                img[i, j] = val

        img /= img.max()
    else:
        raise ValueError(map_name)
    img = img.astype(np.float32)
    return img


def create_uvsampler(uv_mapping_path='data/uv_mappings.txt', tex_size=2):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T*T x 2
    """
    alpha = np.arange(tex_size, dtype=np.float32) / (tex_size - 1)
    beta = np.arange(tex_size, dtype=np.float32) / (tex_size - 1)
    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])  # T*2 x 2

    obj_info = load_obj(uv_mapping_path)

    vts = obj_info['vts']
    vts[:, 1] = 1 - vts[:, 1]
    faces_vts = obj_info['faces_vts']

    # F x 3 x 2
    f2vts = vts[faces_vts]

    # print(f2vts.shape, f2vts.max(), f2vts.min())

    # Compute alpha, beta (this is the same order as NMR)
    v2 = f2vts[:, 2]  # (nf, 2)
    v0v2 = f2vts[:, 0] - f2vts[:, 2]  # (nf, 2)
    v1v2 = f2vts[:, 1] - f2vts[:, 2]  # (nf, 2)

    # F x 2 x T*2
    samples = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 2, 1)
    samples = np.clip(samples, a_min=0.0, a_max=1.0)

    # F x T*2 x 2 points on the sphere
    uv = np.transpose(samples, (0, 2, 1))

    # uv = uv.reshape(-1, tex_size, tex_size, 2)
    # normalize to [-1, 1]
    uv = uv * 2 - 1

    return uv


def vertices_to_faces(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 2]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 2]
    """
    bs, nv = vertices.shape[:2]
    device = vertices.device

    # print(vertices.shape, faces.shape)
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 2))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def faces_to_sampler(coords, faces):
    """
    :param coords: [T*T, 3]
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


class UVImageModel(nn.Module):
    def __init__(self, uv, image_size):
        super(UVImageModel, self).__init__()

        # (1, 3, image_size, image_size)
        # self.weight = nn.Parameter(torch.zeros(1, 3, image_size, image_size) - 1.0)
        self.weight = nn.Parameter(torch.zeros(1, 3, image_size, image_size) - 1.0)

        # (f, t, t, 2)
        self.f, self.t = uv.shape[:2]
        # (1, f, t*t, 2)
        uv = uv.reshape(1, self.f, self.t * self.t, 2)
        self.uv = torch.FloatTensor(uv).cuda()

    def forward(self):
        uv_image = torch.tanh(self.weight)
        texture = F.grid_sample(uv_image, self.uv)
        # (1,3,f,t,t)
        texture = texture.view(1, 3, self.f, self.t, self.t)
        # (1,f,t,t,3)
        texture = texture.permute(0, 2, 3, 4, 1)

        return texture

    def get_uv_image(self):
        return torch.tanh(self.weight)
        # return 2 * torch.sigmoid(self.weight) - 1


def compute_uv_image(uv, texture, uv_size=224):
    """
    :param uv: (f, t, t, 2)
    :param texture: torch.Tensor [f, t, t, 3]
    :param uv_size: int, default is 224
    :return: uv_image (3,h,w) rgb(-1,1)
    """
    with torch.enable_grad():
        uv_image_model = UVImageModel(uv, image_size=uv_size).cuda()
        opt = torch.optim.Adam(uv_image_model.parameters(), lr=1e-2)
        for epoch in range(2000):
            pred_texture = uv_image_model()

            loss = ((pred_texture - texture) ** 2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

            # if epoch % 10 == 0:
            #     print(epoch, loss.item())

    return uv_image_model.get_uv_image()[0]
