import numpy as np

from utils.util import load_pickle_file


data = load_pickle_file('pretrains/smpl_model.pkl')

# 13776 x 3
faces = data['f']


def face_to_graph(faces):
    graph = {}
    verts_faces_set = {}

    def add_to_set(vi, fid):
        if vi not in verts_faces_set:
            verts_faces_set[vi] = set()
        verts_faces_set[vi].add(fid)

    def add_edge(start, edges):
        if start not in graph:
            graph[start] = set()

        graph[start] |= edges

    # 1. compute verts faces set
    print('compute verts faces set')
    for fid in range(faces.shape[0]):
        for vi in faces[fid]:
            # v0, v1, v2
            add_to_set(vi, fid)

    # 2. build graph
    print('build graph')
    for fid in range(faces.shape[0]):
        for vi in faces[fid]:
            connected_faces = verts_faces_set[vi]
            add_edge(fid, connected_faces)
            # print(fid, vi, len(connected_faces))

        # print(fid, '->', graph[fid])

    for fid in graph:
        graph[fid] = list(graph[fid])

        print(fid, '->', graph[fid])

    return graph


graph = face_to_graph(faces)

