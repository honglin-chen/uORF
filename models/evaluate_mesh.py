import numpy as np
import skimage.measure as measure
import mcubes
import h5py
from scipy.spatial.transform import Rotation as R
import torch

toggle_yz = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
])

toggle_xz = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
])

def get_surface_points_from_voxel(V, threshold, num):
    vtx, faces, _, _ = measure.marching_cubes_lewiner(V, threshold)
    d = 5e-7 / (V.shape[0] ** 2)
    pts = get_surface_points_from_mesh(vtx, faces, d, num)

    pts = normalize_points(pts)
    return pts, vtx, faces

def get_surface_points_from_mesh(vtx, faces, d, num):

    points = _sample_ptcld_vf(vtx, faces, d)

    while len(points) < num:
        d *= 2
        points = _sample_ptcld_vf(vtx, faces, d)
    idx = np.arange(len(points))
    np.random.shuffle(idx)
    idx = idx[:num]
    return points[idx, :]

def _sample_ptcld_vf(veticies, faces, d=10):
    area = 0
    triangle = veticies[faces]
    density = d
    pt_num_total = 0
    ptnum_list = list()

    for tr in triangle:
        tr_a = np.linalg.norm(np.cross(tr[1] - tr[0], tr[2] - tr[0])) / 2
        area += tr_a
        ptnum = max(int(tr_a * density), 1)
        pt_num_total += ptnum
        ptnum_list.append(ptnum)

    point = np.zeros([pt_num_total, 3], np.float32)
    cnt = 0
    for idx_tr, tr in enumerate(triangle):
        ptnum = ptnum_list[idx_tr]
        sqrt_r1 = np.sqrt(np.random.rand(ptnum))
        r2 = np.random.rand(ptnum)
        pts = np.outer(1 - sqrt_r1, tr[0]) + np.outer((sqrt_r1) * (1 - r2), tr[1]) + np.outer(r2 * sqrt_r1, tr[2])
        point[cnt:cnt + ptnum, :] = pts
        cnt += ptnum
    return point


def load_list_from_file(file_path):
    with open(file_path) as f:
        lines = f.read().split('\n')
    while lines.count('') > 0:
        lines.remove('')
    return lines


def crop_voxel(voxel, crop_len=0, max_value=1.0):
    """
    load voxel and crop a few outer voxels
    """
    assert voxel.shape[0] == voxel.shape[1] and voxel.shape[0] == voxel.shape[2]
    voxel = np.array(voxel, dtype=float) / max_value
    pred_len = voxel.shape[0]
    voxel = voxel[crop_len:pred_len - crop_len,
                  crop_len:pred_len - crop_len, crop_len:pred_len - crop_len]
    return voxel


def normalize_points(points):
    # normalize point clouds: set center of bounding box to origin, longest side to 1
    bound_l = np.min(points, axis=0)
    bound_h = np.max(points, axis=0)
    points = points - (bound_l + bound_h) / 2
    points = points / (bound_h - bound_l).max()

    points = torch.tensor(points)
    return points

def transform_vertices(pts, rotations, trans, scale):
    rot = R.from_quat(rotations).as_matrix()
    pts = scale[None] * pts
    transformed_pts = np.matmul(rot, pts.T).T + np.expand_dims(trans, axis=0)

    return transformed_pts

def load_gt_mesh_from_hdf(path, frame='0005', num_objects=3, seg_colors=None, seg_masks=None):
    with h5py.File(path, 'r') as f:
        obj_vertices = []
        obj_faces = []
        scene_vertices = []
        scene_faces = []

        if seg_colors is not None:
            static_obj_seg_colors = torch.tensor(f["static"]['object_segmentation_colors'][:])
            match = (seg_colors.unsqueeze(2) == static_obj_seg_colors.unsqueeze(1)).all(-1)[...,0]
            assert match.sum() == seg_colors.shape[0] == static_obj_seg_colors.shape[0]
            match_idx = torch.nonzero(match)[:, 1]
        else:
            match_idx = range(num_objects)

        count = 0
        for obj_idx in match_idx:
            vertices = f["static"]["mesh"][f"vertices_{obj_idx}"][:]
            faces = f["static"]["mesh"][f"faces_{obj_idx}"][:]

            scale = np.array(f["static"]["scale"][:])[obj_idx]
            scale_factor = np.array(f["frames"][frame]["objects"]["scale_factor"][:])[obj_idx]
            scale *= scale_factor
            rotation = np.array(f["frames"][frame]["objects"]["rotations"][:])[obj_idx]
            translation = np.array(f["frames"][frame]["objects"]["positions"][:])[obj_idx]

            vertices = transform_vertices(vertices, rotation, translation, scale)

            vertices = np.matmul(toggle_yz, vertices.T).T
            faces = np.matmul(toggle_yz, faces.T).T
            scene_vertices.append(vertices)
            scene_faces.append(faces + count)
            obj_vertices.append(vertices)
            obj_faces.append(faces)

            count += vertices.shape[0]

        scene_vertices = np.concatenate(scene_vertices)
        scene_faces = np.concatenate(scene_faces)

        return obj_vertices, obj_faces, scene_vertices, scene_faces

def compute_mesh_from_voxel(voxel, threshold):
    # voxel is a numpy array with shape [K, H, W, D], where K is the number of objects
    obj_vertices = []
    obj_faces = []
    scene_vertices = []
    scene_faces = []

    count = 0
    for obj_id in range(voxel.shape[0]):
        vtx, faces, _, _ = measure.marching_cubes_lewiner(voxel[obj_id], threshold)
        # vtx, faces, _, _ = mcubes.marching_cubes(mcubes.smooth(voxel[obj_id]), threshold)

        # transform the axis so that it aligns with the GT pointcloud
        vtx = np.matmul(toggle_xz, vtx.T).T
        vtx[:,2] = -vtx[:, 2] # invert yz

        faces = np.matmul(toggle_xz, faces.T).T.astype(np.uint64)
        scene_vertices.append(vtx)
        scene_faces.append(faces + count)
        obj_vertices.append(vtx)
        obj_faces.append(faces)
        count += vtx.shape[0]

    scene_vertices = np.concatenate(scene_vertices)
    scene_faces = np.concatenate(scene_faces)

    return obj_vertices, obj_faces, scene_vertices, scene_faces


def cd_dis(xyz1, xyz2, batch_size):
    assert xyz1.shape[1] == xyz2.shape[1] and xyz1.shape[2] == 3 and xyz2.shape[2] == 3
    all_scores = list()
    for idx in range(int(math.ceil(xyz1.shape[0] / batch_size))):
        indl = idx * batch_size
        indh = min(xyz1.shape[0], (idx + 1) * batch_size)
        scores = _cd_dis(xyz1[indl:indh].copy(), xyz2[indl:indh].copy())
        all_scores.append(scores)
    all_scores = np.concatenate(all_scores)
    return all_scores

if __name__ == '__main__':

    v1 = np.random.rand(64, 64, 64)
    v2 = np.random.rand(64, 64, 64)

    get_surface_points(v1, threshold=0.1, num=1024)