import numpy as np


from utils.util import load_pickle_file


# def bin2int(bits):
#     total = 0
#     for shift, j in enumerate(bits[::-1]):
#         if j:
#             total += 1 << shift
#     return total
#
#
# print(bin2int(np.array([-1, -1, -1, -1])))

data = load_pickle_file('pretrains/smpl_model.pkl')

# 6890 x 24
weights = data['weights']
faces = data['f']


def get_vertices_joints(weights):

    blend_num = 4

    verts_joints = []

    for num_v, w in enumerate(weights):
        joint_ids = np.nonzero(w)[0]
        # print('#v {}, blend ids = {}, num = {}'.format(num_v, joint_ids, len(joint_ids)))

        assert len(joint_ids) == blend_num

        verts_joints.append(joint_ids)

    verts_joints = np.stack(verts_joints, axis=0)
    print(verts_joints.shape)

    return verts_joints


def check_faces_joints(verts_joints, faces):
    face_verts_joints = verts_joints[faces]
    print(face_verts_joints.shape, faces.shape)

    for nf in range(face_verts_joints.shape[0]):
        verts_joints = face_verts_joints[nf]

        # verts_joints -= verts_joints[0:1, :]
        # print(np.sum(verts_joints))


verts_joints = get_vertices_joints(weights)
check_faces_joints(verts_joints, faces)




