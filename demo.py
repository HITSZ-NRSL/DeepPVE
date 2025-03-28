import os
import sys
import random
import numpy as np
import open3d as o3d
import trimesh
import torch
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from collections import abc as container_abcs
from model import HPRNet


OMNI_OBJECT_SCALE = 0.001


class BenchmarkDataAutoGenerator(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.num_points = 20000
        self.num_test_meshes = 600
        self.num_test_viewpoints = 300
        self.ray_cast_tolerance = 0.001

        f_test = open(os.path.join(self.root_dir, 'test_split.txt'), 'r')
        self.test_filepaths = []
        for line in f_test:
            object_name = line.strip()
            object_type = object_name[:-4]
            obj_mesh_path = os.path.join(root_dir, 'raw', 'raw_scans', object_type, object_name, 'Scan', 'Scan_simplified.obj')
            obj_pcd_path = os.path.join(root_dir, 'raw', 'raw_scans', object_type, object_name, 'Scan', 'Scan_sampled.ply')
            if not os.path.exists(obj_mesh_path) or not os.path.exists(obj_pcd_path):
                print('File not found:', obj_mesh_path, obj_pcd_path)
                continue
            self.test_filepaths.append((obj_mesh_path, obj_pcd_path))
        f_test.close()

        self.camera_views = self.generate_grasp_views_by_np(self.num_test_viewpoints)

    @staticmethod
    def generate_grasp_views_by_np(N=300, phi=(np.sqrt(5) - 1) / 2, center=np.zeros(3), r=1):
        """ View sampling on a unit sphere using Fibonacci lattices.
            Ref: https://arxiv.org/abs/0912.4540

            Input:
                N: [int]
                    number of sampled views
                phi: [float]
                    constant for view coordinate calculation, different phi's bring different distributions, default: (sqrt(5)-1)/2
                center: [np.ndarray, (3,), np.float32]
                    sphere center
                r: [float]
                    sphere radius

            Output:
                views: [np.ndarray, (N,3)]
                    sampled view coordinates
        """
        views = []
        for i in range(N):
            zi = (2 * i + 1) / N - 1
            xi = np.sqrt(1 - zi ** 2) * np.cos(2 * i * np.pi * phi)
            yi = np.sqrt(1 - zi ** 2) * np.sin(2 * i * np.pi * phi)
            views.append([xi, yi, zi])
        views = r * np.array(views) + center
        return views.astype(np.float32)

    def get_pcd_mesh(self, mesh_idx):
        pcd = o3d.io.read_point_cloud(self.test_filepaths[mesh_idx][1])
        pcd = pcd.scale(OMNI_OBJECT_SCALE, [0, 0, 0])
        mesh_ = trimesh.load_mesh(self.test_filepaths[mesh_idx][0])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(mesh_.faces)
        mesh = mesh.scale(OMNI_OBJECT_SCALE, [0, 0, 0])
        return pcd, mesh

    def get_data_label(self, mesh_idx, viewpoint_idx):
        """
        Get data and label for a specific mesh and viewpoint.
        :param mesh_idx: int
        :param viewpoint_idx: int
        :return: np.ndarray, (N,3), np.ndarray, (N,), bool, True for visible, False for invisible
        """
        mesh_ = trimesh.load_mesh(self.test_filepaths[mesh_idx][0])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(mesh_.faces)
        mesh = mesh.scale(OMNI_OBJECT_SCALE, [0, 0, 0])
        pcd = o3d.io.read_point_cloud(self.test_filepaths[mesh_idx][1])
        pcd = pcd.scale(OMNI_OBJECT_SCALE, [0, 0, 0])
        points = np.asarray(pcd.points).astype(np.float32)
        sampled_camera_view = self.camera_views[viewpoint_idx]
        points_diameter = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
        absolute_camera_location = sampled_camera_view * points_diameter / 2 + np.random.uniform(0.05, 0.2)
        visible_mask = self.generate_hpr_label(points, mesh, absolute_camera_location)
        points = points - absolute_camera_location
        return points, visible_mask

    def generate_hpr_label(self, points, mesh, camera_locations):
        """
        Generate hidden points removal label for points, using ray casting on mesh.
        :param points: np.ndarray, (N,3)
        :param mesh: open3d.geometry.TriangleMesh
        :param camera_locations: np.ndarray, (3,)
        :return: np.ndarray, (N,), bool, True for visible, False for invisible
        """
        mesh_ = o3d.t.geometry.TriangleMesh(np.asarray(mesh.vertices).astype(np.float32), np.asarray(mesh.triangles).astype(np.int32))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_)
        points = points.astype(np.float32)
        camera_locations = camera_locations.astype(np.float32)
        distances = np.linalg.norm(points - camera_locations, axis=1, keepdims=True)  # [N,1]
        rays = np.zeros((points.shape[0], 6), dtype=np.float32)  # [N,6]
        rays[:, :3] = camera_locations
        rays[:, 3:] = (points - camera_locations) / distances
        cast_results = scene.cast_rays(o3d.core.Tensor(rays))
        hit_distances = cast_results['t_hit'].numpy()  # [N,]
        visible_mask = hit_distances + self.ray_cast_tolerance > distances.squeeze()  # [N,]
        return visible_mask.astype(np.float32)

    def global_density_variation(self, points, visible_mask, target_point_num):
        """
        down sample points to target_point_num
        :param points: np.ndarray, (N,3)
        :param visible_mask: np.ndarray, (N,), bool
        :param target_point_num: int
        :return:
        """
        assert points.shape[0] >= target_point_num, 'target_point_num should be smaller than points.shape[0]'
        remain_idxs = np.random.choice(points.shape[0], target_point_num, replace=False)
        return points[remain_idxs].copy(), visible_mask[remain_idxs].copy()

    def local_density_variation(self, points, visible_mask, downsample_ratio):
        """
        compute the center, randomly split the points into two parts, down sample one part
        :param points: np.ndarray, (N,3)
        :param visible_mask: np.ndarray, (N,), bool
        :param downsample_ratio: float, 0~1
        :return: np.ndarray, (N,3), np.ndarray, (N,), bool, True for visible, False for invisible
        """
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        sample_mask = distances <= np.percentile(distances, 50)
        unsample_points = points[~sample_mask].copy()
        unsample_visible_mask = visible_mask[~sample_mask].copy()
        sample_points = points[sample_mask].copy()
        split_visible_mask = visible_mask[sample_mask].copy()
        downsample_num = np.round(points.shape[0] * (0.5 + 0.5 * downsample_ratio)).astype(int) - unsample_points.shape[0]
        remain_idxs = np.random.choice(len(sample_points), downsample_num, replace=False)
        sample_points = sample_points[remain_idxs].copy()
        split_visible_mask = split_visible_mask[remain_idxs].copy()
        points_ = np.concatenate([sample_points, unsample_points], axis=0)
        visible_mask_ = np.concatenate([split_visible_mask, unsample_visible_mask], axis=0)
        return points_, visible_mask_

    def noise_variation_type1(self, points, visible_mask, noise_mu):
        """
        add noise to the points
        :param points: np.ndarray, (N,3)
        :param visible_mask: np.ndarray, (N,), bool
        :param noise_mu: float
        :return: np.ndarray, (N,3), np.ndarray, (N,), bool, True for visible, False for invisible
        """
        # noise type 1
        diameter = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        points_ = (points + noise_mu * diameter * normals * np.random.randn(points.shape[0], 1)).astype(np.float32)
        return points_, visible_mask

    def noise_variation_type2(self, points, visible_mask, noise_mu):
        """
        add noise to the points
        :param points: np.ndarray, (N,3)
        :param visible_mask: np.ndarray, (N,), bool
        :param noise_mu: float
        :return: np.ndarray, (N,3), np.ndarray, (N,), bool, True for visible, False for invisible
        """
        # noise type 2
        unit_directions = np.random.uniform(-1, 1, (points.shape[0], 3))
        unit_directions /= np.linalg.norm(unit_directions, axis=1, keepdims=True)
        points_ = (points * (1 + np.random.uniform(0, noise_mu) * unit_directions)).astype(np.float32)
        return points_, visible_mask


def deephpr(points, viewpoint, labels):
    VOXEL_SIZE = 0.001
    def minkowski_collate_fn(list_data):
        """
        :param list_data: dict(points, labels)
        :return: dict(points, normals, coors, feats, quantize2original)
        """
        list_data_ = []
        for d in list_data:
            d_ = {}
            coors = d[0] - np.min(d[0], axis=0, keepdims=True)
            coors = np.floor(coors / VOXEL_SIZE).astype(np.int32)
            d_["points"] = d[0].copy()
            d_["coors"] = coors.copy()
            d_["feats"] = d[0].copy()
            d_["labels"] = d[1].copy()
            list_data_.append(d_)
        coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate([d["coors"] for d in list_data_],
                                                                                  [d["feats"] for d in list_data_],
                                                                                  [d["labels"] for d in list_data_])
        coordinates_batch, features_batch, labels_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch.float(), features_batch, labels_batch.int(), return_index=True, return_inverse=True)

        res = {"coors": coordinates_batch,
               "feats": features_batch,
               "labels_quant": labels_batch,
               "quantize2original": quantize2original
               }

        def collate_fn_(batch):
            if type(batch[0]).__module__ == 'numpy' and isinstance(batch[0], np.ndarray):
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
            elif type(batch[0]).__module__ == 'numpy' and not isinstance(batch[0], np.ndarray):
                return torch.stack([torch.from_numpy(np.array(b)) for b in batch], 0)
            elif isinstance(batch[0], container_abcs.Sequence):
                return [[torch.from_numpy(sample) for sample in b] for b in batch]
            elif isinstance(batch[0], container_abcs.Mapping):
                for key in batch[0]:
                    if key == 'coors' or key == 'feats':
                        continue
                    res[key] = collate_fn_([d[key] for d in batch])
                return res
        res = collate_fn_(list_data_)
        return res

    points = points - viewpoint
    batch_data = minkowski_collate_fn([(points.astype(np.float32), labels)])
    model = HPRNet().cuda()
    model.load_state_dict(torch.load('weights/hprnet_best.pth')['model_state_dict'])
    model.eval()
    with torch.no_grad():
        for key in batch_data:
            batch_data[key] = batch_data[key].cuda()
        logits = model(batch_data)['logits']
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        # print(pred.sum(), batch_data['coors'].shape[0], logits.shape)
    visible_mask = pred[batch_data['quantize2original'].cpu()]
    return visible_mask


def test_armadillo():
    data_generator = BenchmarkDataAutoGenerator(root_dir='/home/myubuntu/FastData/OpenXD-OmniObject3D-New')
    armadillo_path = 'assets/Armadillo.ply'
    mesh = o3d.io.read_triangle_mesh(armadillo_path)
    mesh.compute_vertex_normals()
    mesh = mesh.scale(OMNI_OBJECT_SCALE, [0, 0, 0])
    pcd = mesh.sample_points_poisson_disk(data_generator.num_points)
    viewpoint_idx = np.random.randint(0, 300)
    mesh.compute_vertex_normals()
    points = np.asarray(pcd.points)
    diameter = np.linalg.norm(np.max(np.asarray(pcd.points), axis=0) - np.min(np.asarray(pcd.points), axis=0))
    extra_distance = np.random.uniform(0.05, 0.3)
    viewpoint = data_generator.camera_views[viewpoint_idx] * diameter / 2 + extra_distance
    visible_mask = data_generator.generate_hpr_label(points, mesh, viewpoint)
    points_ori = points.copy()
    visible_mask_ori = visible_mask.copy()

    # global density variation, 16000 points
    points, visible_mask = data_generator.global_density_variation(points_ori, visible_mask_ori, 12000)
    points, visible_mask = data_generator.noise_variation_type1(points, visible_mask, 0.010)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((points.shape[0], 3))
    colors[:, 0] = 1  # invisible: red
    colors[visible_mask == 1] = [0, 1, 0]  # visible: green
    pcd.colors = o3d.utility.Vector3dVector(colors)

    extinsic = np.eye(4)
    z_axis = np.mean(points, axis=0) - viewpoint
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.array([-z_axis[1], z_axis[0], 0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    extinsic[:3, 0] = x_axis
    extinsic[:3, 1] = y_axis
    extinsic[:3, 2] = z_axis
    extinsic[:3, 3] = viewpoint
    intrinsic = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]])
    extinsic = np.linalg.inv(extinsic)
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(640, 480, intrinsic, extinsic, 0.02)
    csys = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([mesh, cameraLines])
    o3d.visualization.draw_geometries([pcd, cameraLines], window_name='GT')

    # DeepHPR
    deephpr_visible_mask = deephpr(points, viewpoint, visible_mask)
    acc_deephpr = np.sum(deephpr_visible_mask == visible_mask) / points.shape[0]
    colors = np.zeros((points.shape[0], 3))
    colors[:, 0] = 1  # invisible: red
    colors[deephpr_visible_mask == 1] = [0, 1, 0]  # visible: green
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd, cameraLines], window_name='DeepHPR, acc: {:.4f}'.format(acc_deephpr))
    print('DeepHPR acc:', acc_deephpr)

if __name__ == '__main__':
    test_armadillo()
