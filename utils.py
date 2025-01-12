import torch
import kaolin as kal
import kaolin.ops.mesh
import clip
import numpy as np
from torchvision import transforms
from pathlib import Path
from collections import Counter
from Normalization import MeshNormalizer

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def get_camera_from_view2(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj

def get_texture_map_from_color(mesh, color, H=224, W=224):
    num_faces = mesh.faces.shape[0]
    texture_map = torch.zeros(1, H, W, 3).to(device)
    texture_map[:, :, :] = color
    return texture_map.permute(0, 3, 1, 2)


def get_face_attributes_from_color(mesh, color):
    num_faces = mesh.faces.shape[0]
    face_attributes = torch.zeros(1, num_faces, 3, 3).to(device)
    face_attributes[:, :, :] = color
    return face_attributes

# mesh coloring helpers

def color_mesh(pred_class, sampled_mesh, colors):
    pred_rgb = segment2rgb(pred_class, colors)
    sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    MeshNormalizer(sampled_mesh)()
    return pred_rgb
'''
def color_mesh(pred_class, sampled_mesh, colors):
    """
    Colors the mesh based on predicted class and colors.
    Args:
        pred_class: [N, 2] - predicted class probabilities for each vertex
        sampled_mesh: Mesh - mesh to be colored
        colors: [2, 3] - RGB colors for each class
    """
    pred_rgb = segment2rgb(pred_class, colors)  # Use the original colors tensor
    sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces
    ).squeeze(0)
    return pred_rgb  # Return colored vertices
'''

def segment2rgb(pred_class, colors):
    pred_rgb = torch.zeros(pred_class.shape[0], 3).to(device)
    for class_idx, color in enumerate(colors):
        pred_rgb += torch.matmul(pred_class[:,class_idx].unsqueeze(1), color.unsqueeze(0))
        
    return pred_rgb
    
'''
def segment2rgb(pred_class, colors):
    """
    Converts predicted class to RGB colors.
    Args:
        pred_class: [N, 2] - predicted class probabilities for each vertex
        colors: [2, 3] - RGB colors for each class
    """
    max_idx = torch.argmax(pred_class, 1, keepdim=True)  # Get class indices
    pred_rgb = torch.zeros(pred_class.shape[0], 3).to(device)  # Initialize RGB tensor

    for idx in range(pred_class.shape[0]):
        pred_rgb[idx] = colors[max_idx[idx].item()]  # Use the correct index from max_idx and original colors tensor

    return pred_rgb  # Return RGB colors
'''

def assign_colors(pred_class, colors, device):
    """
    Assign colors to points based on their probabilities.

    Args:
        pred_class (torch.Tensor): Tensor of shape [N, 2], where each row contains 
                                   [p_highlighted, p_not_highlighted].
        colors (torch.Tensor): Tensor of shape [2, 3], where the first row is "highlighter" color 
                               and the second row is "gray" color.
        device (torch.device): Device to which the tensors are moved (e.g., "cpu" or "cuda").

    Returns:
        torch.Tensor: A tensor of shape [N, 3] containing the RGB colors for each point.
    """
    # Ensure the probabilities are normalized (if not already)
    pred_class = pred_class / pred_class.sum(dim=1, keepdim=True)
    
    # Extract probabilities for highlighter and gray
    p_highlighter = pred_class[:, 0].unsqueeze(1)  # Shape: [N, 1]
    p_gray = pred_class[:, 1].unsqueeze(1)         # Shape: [N, 1]
    
    # Compute weighted colors
    colors_highlighter = colors[0].unsqueeze(0)  # Shape: [1, 3]
    colors_gray = colors[1].unsqueeze(0)        # Shape: [1, 3]
    
    # Weighted sum of the colors based on probabilities
    point_colors = p_highlighter * colors_highlighter + p_gray * colors_gray  # Shape: [N, 3]

    return point_colors.to(device)

