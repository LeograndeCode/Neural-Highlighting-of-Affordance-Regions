import torch.nn as nn
import numpy as np
import torch
import subprocess
import sys 
from torchvision import transforms

RESOLUTION = 128
TRANS = -1.6


# Specify the version of matplotlib you want to install
required_version = '3.7.1'  # Replace with the desired version

# Use subprocess to run pip and install the required version
subprocess.check_call([sys.executable, "-m", "pip", "install", f"matplotlib=={required_version}"])

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 

def point_cloud_render_to_tensor(point_cloud, image_size=224, views=6):
    """
    Render point cloud from multiple views and return a tensor of shape [6, 3, 224, 224].
    """
    # Extract coordinates and colors
    coordinates = point_cloud[:, :3].cpu().numpy()  # [x, y, z]
    colors = point_cloud[:, 3:].cpu().numpy()       # [r, g, b]

    # Ensure colors are in the range [0, 1]
    colors = np.clip(colors, 0, 1)

    # Normalize coordinates for better visualization
    coordinates = (coordinates - np.min(coordinates, axis=0)) / np.ptp(coordinates, axis=0)
    
    renders = []

    plt.ioff()

    # Create multiple views by rotating the point cloud
    for i, angle in enumerate(np.linspace(0, 360, views, endpoint=False)):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        
        # Rotate point cloud around z-axis
        rotation_matrix = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), 0],
            [0, 0, 1]
        ])
        rotated_coords = coordinates @ rotation_matrix.T

        # Plot point cloud with correct colors
        ax.scatter(rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2],
                   c=colors, s=1, marker='o')
        ax.axis("off")
        ax.set_box_aspect([1, 1, 1])
        
        # Save plot to a buffer
        fig.canvas.draw()
        
        #img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        #img = np.frombuffer(fig.canvas.tostring_rgba(), dtype=np.uint8)
        #img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        #img = img[:, :, :3] 

        #img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        #img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB, handling potential alpha issues
        #img = img[:, :, :3].copy()  # Keep only RGB channels

        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)

        # Resize to [224, 224] and convert to tensor
        img_resized = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        img_resized = torch.nn.functional.interpolate(img_resized.unsqueeze(0), size=(image_size, image_size))[0]
        
        renders.append(img_resized)

    # Stack renders into tensor of shape [6, 3, 224, 224]
    render_tensor = torch.stack(renders)
    plt.ion()
    return render_tensor

'''
def point_cloud_render_to_tensor(point_cloud, image_size=224, views=6):
    """
    Render point cloud from multiple views and return a tensor of shape [6, 3, 224, 224].
    """
    # Extract coordinates and colors
    coordinates = point_cloud[:, :3].cpu().numpy()  # [x, y, z]
    colors = point_cloud[:, 3:].cpu().numpy()       # [r, g, b]

    # Normalize coordinates for better visualization
    coordinates = (coordinates - np.min(coordinates, axis=0)) / np.ptp(coordinates, axis=0)
    
    renders = []

    # Create multiple views by rotating the point cloud
    for i, angle in enumerate(np.linspace(0, 360, views, endpoint=False)):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        
        # Rotate point cloud around z-axis
        rotation_matrix = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), 0],
            [0, 0, 1]
        ])
        rotated_coords = coordinates @ rotation_matrix.T

        #Check if rotated_coords is empty or has invalid shape
        if rotated_coords.size == 0 or rotated_coords.ndim != 2:
            print(f"Warning: rotated_coords has invalid shape: {rotated_coords.shape}. Skipping this view.")
            plt.close(fig)
            continue

        # Plot point cloud
        ax.scatter(rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2],
                   c=colors, s=1, marker='o')
        ax.axis("off")
        ax.set_box_aspect([1, 1, 1])
        
        # Save plot to a buffer
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 
        # Reverse the order of color channels (BGRA to RGB)
        #img = img[:, :, :3][:, :, ::-1].copy() 
        plt.close(fig)

        # Resize to [224, 224] and convert to tensor
        img_resized = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_resized = torch.nn.functional.interpolate(img_resized.unsqueeze(0), size=(image_size, image_size))[0]
        
        renders.append(img_resized)

    # Stack renders into tensor of shape [6, 3, 224, 224]
    render_tensor = torch.stack(renders)
    return render_tensor
'''

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """

    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach()*0
    one = zero.detach()+1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat


def distribute(depth, _x, _y, size_x, size_y, image_height, image_width):
    """
    Distributes the depth associated with each point to the discrete coordinates (image_height, image_width) in a region
    of size (size_x, size_y).
    :param depth:
    :param _x:
    :param _y:
    :param size_x:
    :param size_y:
    :param image_height:
    :param image_width:
    :return:
    """

    assert size_x % 2 == 0 or size_x == 1
    assert size_y % 2 == 0 or size_y == 1
    batch, _ = depth.size()
    epsilon = torch.tensor([1e-12], requires_grad=False, device=depth.device)
    _i = torch.linspace(-size_x / 2, (size_x / 2) - 1, size_x, requires_grad=False, device=depth.device)
    _j = torch.linspace(-size_y / 2, (size_y / 2) - 1, size_y, requires_grad=False, device=depth.device)

    extended_x = _x.unsqueeze(2).repeat([1, 1, size_x]) + _i  # [batch, num_points, size_x]
    extended_y = _y.unsqueeze(2).repeat([1, 1, size_y]) + _j  # [batch, num_points, size_y]

    extended_x = extended_x.unsqueeze(3).repeat([1, 1, 1, size_y])  # [batch, num_points, size_x, size_y]
    extended_y = extended_y.unsqueeze(2).repeat([1, 1, size_x, 1])  # [batch, num_points, size_x, size_y]

    extended_x.ceil_()
    extended_y.ceil_()

    value = depth.unsqueeze(2).unsqueeze(3).repeat([1, 1, size_x, size_y])  # [batch, num_points, size_x, size_y]

    # all points that will be finally used
    masked_points = ((extended_x >= 0)
                     * (extended_x <= image_height - 1)
                     * (extended_y >= 0)
                     * (extended_y <= image_width - 1)
                     * (value >= 0))

    true_extended_x = extended_x
    true_extended_y = extended_y

    # to prevent error
    extended_x = (extended_x % image_height)
    extended_y = (extended_y % image_width)

    # [batch, num_points, size_x, size_y]
    distance = torch.abs((extended_x - _x.unsqueeze(2).unsqueeze(3))
                         * (extended_y - _y.unsqueeze(2).unsqueeze(3)))
    weight = (masked_points.float()
          * (1 / (value + epsilon)))  # [batch, num_points, size_x, size_y]
    weighted_value = value * weight

    weight = weight.view([batch, -1])
    weighted_value = weighted_value.view([batch, -1])

    coordinates = (extended_x.view([batch, -1]) * image_width) + extended_y.view(
        [batch, -1])
    coord_max = image_height * image_width
    true_coordinates = (true_extended_x.view([batch, -1]) * image_width) + true_extended_y.view(
        [batch, -1])
    true_coordinates[~masked_points.view([batch, -1])] = coord_max
    weight_scattered = torch.zeros(
        [batch, image_width * image_height],
        device=depth.device).scatter_add(1, coordinates.long(), weight)

    masked_zero_weight_scattered = (weight_scattered == 0.0)
    weight_scattered += masked_zero_weight_scattered.float()

    weighed_value_scattered = torch.zeros(
        [batch, image_width * image_height],
        device=depth.device).scatter_add(1, coordinates.long(), weighted_value)

    return weighed_value_scattered,  weight_scattered


def points2depth(points, image_height, image_width, size_x=4, size_y=4):
    """
    :param points: [B, num_points, 3]
    :param image_width:
    :param image_height:
    :param size_x:
    :param size_y:
    :return:
        depth_recovered: [B, image_width, image_height]
    """

    epsilon = torch.tensor([1e-12], requires_grad=False, device=points.device)
    # epsilon not needed, kept here to ensure exact replication of old version
    coord_x = (points[:, :, 0] / (points[:, :, 2] + epsilon)) * (image_width / image_height)  # [batch, num_points]
    coord_y = (points[:, :, 1] / (points[:, :, 2] + epsilon))  # [batch, num_points]

    batch, total_points, _ = points.size()
    depth = points[:, :, 2]  # [batch, num_points]
    # pdb.set_trace()
    _x = ((coord_x + 1) * image_height) / 2
    _y = ((coord_y + 1) * image_width) / 2

    weighed_value_scattered, weight_scattered = distribute(
        depth=depth,
        _x=_x,
        _y=_y,
        size_x=size_x,
        size_y=size_y,
        image_height=image_height,
        image_width=image_width)

    depth_recovered = (weighed_value_scattered / weight_scattered).view([
        batch, image_height, image_width
    ])

    return depth_recovered


# source: https://discuss.pytorch.org/t/batched-index-select/9115/6
def batched_index_select(inp, dim, index):
    """
    input: B x * x ... x *
    dim: 0 < scalar
    index: B x M
    """
    views = [inp.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def point_fea_img_fea(point_fea, point_coo, h, w):
    """
    each point_coo is of the form (x*w + h). points not in the canvas are removed
    :param point_fea: [batch_size, num_points, feat_size]
    :param point_coo: [batch_size, num_points]
    :return:
    """
    assert len(point_fea.shape) == 3
    assert len(point_coo.shape) == 2
    assert point_fea.shape[0:2] == point_coo.shape

    coo_max = ((h - 1) * w) + (w - 1)
    mask_point_coo = (point_coo >= 0) * (point_coo <= coo_max)
    point_coo *= mask_point_coo.float()
    point_fea *= mask_point_coo.float().unsqueeze(-1)

    print("point fea :")
    print(point_fea)

    point_coo = point_coo / torch.tensor([w, h])  # Normalize coordinates to [0, 1]
    point_coo = point_coo * torch.tensor([w, h])  # Scale to match image size
    point_coo = point_coo.round().long()  # Round to nearest integer and convert to long

    bs, _, fs = point_fea.shape
    point_coo = point_coo.unsqueeze(2).repeat([1, 1, fs])
    img_fea = torch.zeros([bs, h * w, fs], device=point_fea.device).scatter_add(1, point_coo.long(), point_fea)
    print("img_fea shape: ",img_fea.shape)
    print("img_fea",img_fea)
    

    return img_fea


def distribute_img_fea_points(img_fea, point_coord):
    """
    :param img_fea: [B, C, H, W]
    :param point_coord: [B, num_points], each coordinate  is a scalar value given by (x * W) + y
    :return
        point_fea: [B, num_points, C], for points with coordinates outside the image, we return 0
    """
    B, C, H, W = list(img_fea.size())
    img_fea = img_fea.permute(0, 2, 3, 1).view([B, H*W, C])

    coord_max = ((H - 1) * W) + (W - 1)
    mask_point_coord = (point_coord >= 0) * (point_coord <= coord_max)
    mask_point_coord = mask_point_coord.float()
    point_coord = mask_point_coord * point_coord
    point_fea = batched_index_select(
        inp=img_fea,
        dim=1,
        index=point_coord.long())
    point_fea = mask_point_coord.unsqueeze(-1) * point_fea
    return point_fea

def project(points, colors, device, h, w):

      
      with torch.no_grad():
        #points = points.squeeze(0).cpu().numpy()  # Move points to CPU and convert to NumPy array
        #colors = colors.squeeze(0).cpu().numpy() 
        points = points.cpu().numpy() # Convert points to a NumPy array directly
        colors = colors.cpu().numpy()
        
        # Project 3D points to 2D using orthogonal projection
        u = points[:, 0]  # Horizontal coordinate
        v = points[:, 1]  # Vertical coordinate

        # Normalize and scale to image dimensions
        u = ((u - np.min(u)) / (np.max(u) - np.min(u)) * (w - 1)).astype(np.int32)
        v = ((v - np.min(v)) / (np.max(v) - np.min(v)) * (h - 1)).astype(np.int32)
        point_coo = (u * w + v)  # Convert to linear indices
        print(point_coo)

            # Prepare point_fea with RGB colors
        point_fea = torch.tensor(colors, dtype=torch.float32).unsqueeze(0).to(device)  # [batch_size, num_points, feat_size]
        print(point_fea)

            # Convert point_coo to tensor
        point_coo = torch.tensor(point_coo, dtype=torch.int32).unsqueeze(0).to(device)  # [batch_size, num_points]

        point_coo = point_coo.type(torch.float32)

        img_fea = point_fea_img_fea(point_fea, point_coo, h, w)
        print(img_fea)

        img_fea_img = img_fea.view(1, h, w, -1).permute(0, 3, 1, 2).squeeze(1)  # [1, 3, 128, 128]


      return img_fea_img

class PCViews:
    """For creating images from PC based on the view information. Faster as the
    repeated operations are done only once whie initialization.
    """

    def __init__(self):
        _views = np.asarray([
            [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [0, 0, TRANS]],
            [[0, np.pi / 2, np.pi / 2], [0, 0, TRANS]]]
            )
            
        self.num_views = 6

        angle = torch.tensor(_views[:, 0, :]).float().cuda()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float().cuda()
        self.translation = self.translation.unsqueeze(1)




    def get_img(self, points, colors):
        """Get image based on the prespecified specifications.

        Args:
            points (torch.tensor): of size [B, _, 3]
        Returns:
            img (torch.tensor): of size [B * self.num_views, RESOLUTION,
                RESOLUTION]
        """
        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))

        #print(_points)
        print("points shape:")
        print(_points.shape)

        imgs = []
        for i in range(self.num_views):
          #imgs.append(point_fea_img_fea(colors, _points[i,:,:], RESOLUTION, RESOLUTION))
          imgs.append(project(points, colors, "cuda", RESOLUTION, RESOLUTION))

        ret_img = torch.stack(imgs, dim=0)

        img = points2depth(
            points=_points,
            image_height=RESOLUTION,
            image_width=RESOLUTION,
            size_x=1,
            size_y=1,
        )

        return ret_img
    
    def get_img_with_color(self, points, colors):
        """
        Generate depth and color image using the `distribute()` function for color distribution.
    
        Args:
            points (torch.tensor): [num_points, 3] - point cloud coordinates
            colors (torch.tensor): [num_points, 3] - RGB values per point

        Returns:
            img_with_color (torch.tensor): [B * self.num_views, RESOLUTION, RESOLUTION, 3]
        """
        #points = points.unsqueeze(0)
        #colors = colors.unsqueeze(0)
        
        b, num_points, _ = points.shape
        #b, num_points, _ = points.shape
        
        v = self.translation.shape[0]
    
        # Repeat points and colors for all views
        points_repeated = torch.repeat_interleave(points, v, dim=0)
        colors_repeated = torch.repeat_interleave(colors, v, dim=0)
    
        # Transform points for each view
        transformed_points = self.point_transform(
            points=points_repeated,
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1)
        )

        print(transformed_points.shape)


        # Project coordinates to image space
        epsilon = 1e-12
        coord_x = (transformed_points[:, :, 0] / (transformed_points[:, :, 2] + epsilon)) * (RESOLUTION / RESOLUTION)
        coord_y = (transformed_points[:, :, 1] / (transformed_points[:, :, 2] + epsilon))
    
        # Convert to pixel indices
        _x = ((coord_x + 1) * RESOLUTION // 2)
        _y = ((coord_y + 1) * RESOLUTION // 2)
    
        # Distribute colors for each channel (R, G, B)
        color_channels = []
        for i in range(3):  # Iterate over R, G, B channels
            weighed_color_scattered, weight_color_scattered = distribute(
                depth=colors_repeated[:, :, i],
                _x=_x,
                _y=_y,
                size_x=1,
                size_y=1,
                image_height=RESOLUTION,
                image_width=RESOLUTION
            )
            color_channel = (weighed_color_scattered / weight_color_scattered).view(
                b * v, RESOLUTION, RESOLUTION
            )
            color_channels.append(color_channel)
    
        # Stack channels into final color map
        color_map = torch.stack(color_channels, dim=-1)  # Shape: [B * num_views, RESOLUTION, RESOLUTION, 3]

        print(color_map.shape)
        print(color_map)

        return color_map

    @staticmethod
    def point_transform(points, rot_mat, translation):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = points - translation
        return points
