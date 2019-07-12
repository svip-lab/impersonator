from __future__ import division
import os

import torch
import numpy as np
import cv2


import neural_renderer.cuda.load_textures as load_textures_cuda

def load_mtl(filename_mtl):
    '''
    load color (Kd) and filename of textures from *.mtl
    '''
    texture_filenames = {}
    colors = {}
    material_name = ''
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
                if line.split()[0] == 'map_Kd':
                    texture_filenames[material_name] = line.split()[1]
                if line.split()[0] == 'Kd':
                    colors[material_name] = np.array(list(map(float, line.split()[1:4])))
    return colors, texture_filenames


def load_textures(filename_obj, filename_mtl, texture_size):
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices.append([float(v) for v in line.split()[1:3]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces for textures
    faces = []
    material_names = []
    material_name = ''
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            if '/' in vs[0]:
                v0 = int(vs[0].split('/')[1])
            else:
                v0 = 0
            for i in range(nv - 2):
                if '/' in vs[i + 1]:
                    v1 = int(vs[i + 1].split('/')[1])
                else:
                    v1 = 0
                if '/' in vs[i + 2]:
                    v2 = int(vs[i + 2].split('/')[1])
                else:
                    v2 = 0
                faces.append((v0, v1, v2))
                material_names.append(material_name)
        if line.split()[0] == 'usemtl':
            material_name = line.split()[1]
    faces = np.vstack(faces).astype(np.int32) - 1
    faces = vertices[faces]
    faces = torch.from_numpy(faces).cuda()
    faces[1 < faces] = faces[1 < faces] % 1

    colors, texture_filenames = load_mtl(filename_mtl)

    textures = torch.zeros(faces.shape[0], texture_size, texture_size, texture_size, 3, dtype=torch.float32) + 0.5
    textures = textures.cuda()

    #
    for material_name, color in colors.items():
        color = torch.from_numpy(color).cuda()
        for i, material_name_f in enumerate(material_names):
            if material_name == material_name_f:
                textures[i, :, :, :, :] = color[None, None, None, :]

    for material_name, filename_texture in texture_filenames.items():
        filename_texture = os.path.join(os.path.dirname(filename_obj), filename_texture)
        image = cv2.imread(filename_texture).astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # pytorch does not support negative slicing for the moment
        image = image[::-1, :, :]
        image = torch.from_numpy(image.copy()).cuda()
        is_update = (np.array(material_names) == material_name).astype(np.int32)
        is_update = torch.from_numpy(is_update).cuda()
        textures = load_textures_cuda.load_textures(image, faces, textures, is_update)
    return textures


def load_obj(filename_obj, normalization=True, texture_size=4, load_texture=False):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1

    # load textures
    textures = None
    if load_texture:
        for line in lines:
            if line.startswith('mtllib'):
                filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
                textures = load_textures(filename_obj, filename_mtl, texture_size)
        if textures is None:
            raise Exception('Failed to load textures.')

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    if load_texture:
        return vertices, faces, textures
    else:
        return vertices, faces
