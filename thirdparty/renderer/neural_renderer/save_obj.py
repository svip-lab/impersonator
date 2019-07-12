from __future__ import division
import os

import torch
import cv2
import numpy as np


import neural_renderer.cuda.create_texture_image as create_texture_image_cuda


def create_texture_image(textures, texture_size_out=16):
    num_faces, texture_size_in = textures.shape[:2]
    tile_width = int((num_faces - 1.) ** 0.5) + 1
    tile_height = int((num_faces - 1.) / tile_width) + 1
    image = torch.zeros(tile_height * texture_size_out, tile_width * texture_size_out, 3, dtype=torch.float32)
    vertices = torch.zeros((num_faces, 3, 2), dtype=torch.float32)  # [:, :, XY]
    face_nums = torch.arange(num_faces)
    column = face_nums % tile_width
    row = face_nums / tile_width
    vertices[:, 0, 0] = column * texture_size_out
    vertices[:, 0, 1] = row * texture_size_out
    vertices[:, 1, 0] = column * texture_size_out
    vertices[:, 1, 1] = (row + 1) * texture_size_out - 1
    vertices[:, 2, 0] = (column + 1) * texture_size_out - 1
    vertices[:, 2, 1] = (row + 1) * texture_size_out - 1
    image = image.cuda()
    vertices = vertices.cuda()
    textures = textures.cuda()
    image = create_texture_image_cuda.create_texture_image(vertices, textures, image, 1e-5)
    # image = torch.ones_like(image)

    print(image.dtype, image.min(), image.max())

    vertices[:, :, 0] /= (image.shape[1] - 1)
    vertices[:, :, 1] /= (image.shape[0] - 1)

    image = image.detach().cpu().numpy()
    vertices = vertices.detach().cpu().numpy()
    image = image[::-1, ::1]

    return image, vertices


def save_obj(filename, vertices, faces, textures=None):
    assert vertices.ndimension() == 2
    assert faces.ndimension() == 2

    if textures is not None:
        filename_mtl = filename[:-4] + '.mtl'
        filename_texture = filename[:-4] + '.png'
        material_name = 'material_1'
        texture_image, vertices_textures = create_texture_image(textures)
        # imsave(filename_texture, texture_image)

        print(texture_image.dtype, texture_image.max(), texture_image.min())
        texture_image = (texture_image + 1) / 2.0 * 255.0
        texture_image = texture_image.astype(np.uint8)
        cv2.imwrite(filename_texture, texture_image)

    faces = faces.detach().cpu().numpy()

    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('g\n')

        if textures is not None:
            f.write('mtllib %s\n\n' % os.path.basename(filename_mtl))

        for vertex in vertices:
            f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
        f.write('\n')

        if textures is not None:
            for vertex in vertices_textures.reshape((-1, 2)):
                f.write('vt %.8f %.8f\n' % (vertex[0], vertex[1]))
            f.write('\n')

            f.write('usemtl %s\n' % material_name)
            for i, face in enumerate(faces):
                f.write('f %d/%d %d/%d %d/%d\n' % (
                    face[0] + 1, 3 * i + 1, face[1] + 1, 3 * i + 2, face[2] + 1, 3 * i + 3))
            f.write('\n')
        else:
            for face in faces:
                f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))

        f.write('s off\n')
        print('save obj to {}'.format(filename))

    if textures is not None:
        with open(filename_mtl, 'w') as f:
            f.write('newmtl %s\n' % material_name)
            f.write('map_Kd %s\n' % os.path.basename(filename_texture))