# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import numpy as np


# # class TinyNeRF(nn.Module):
# #     """
# #     A lightweight, differentiable Neural Radiance Field (NeRF) implementation.
    
# #     This implementation focuses on:
# #     1. Differentiability (for end-to-end training)
# #     2. Efficiency (for faster training)
# #     3. Simplicity (fewer parameters than full NeRF)
    
# #     The model renders novel views given input images and camera poses.
# #     """
# #     def __init__(self,
# #                  num_encoding_functions=10,
# #                  hidden_dim=128,
# #                  num_layers=4,
# #                  image_height=256,
# #                  image_width=256):
# #         super().__init__()
        
# #         # Positional encoding dimensions
# #         self.num_encoding_functions = num_encoding_functions
# #         self.in_dim_position = 3 + 3 * 2 * num_encoding_functions  # (x,y,z) + positional encoding
# #         self.in_dim_direction = 3 + 3 * 2 * num_encoding_functions  # (d_x,d_y,d_z) + positional encoding
        
# #         # Image dimensions
# #         self.image_height = image_height
# #         self.image_width = image_width
        
# #         # MLP for density prediction
# #         density_layers = []
# #         density_layers.append(nn.Linear(self.in_dim_position, hidden_dim))
# #         density_layers.append(nn.ReLU())
        
# #         for _ in range(num_layers - 2):
# #             density_layers.append(nn.Linear(hidden_dim, hidden_dim))
# #             density_layers.append(nn.ReLU())
            
# #         density_layers.append(nn.Linear(hidden_dim, hidden_dim + 1))  # +1 for density sigma
        
# #         self.density_net = nn.Sequential(*density_layers)
        
# #         # MLP for color prediction
# #         self.color_net = nn.Sequential(
# #             nn.Linear(hidden_dim + self.in_dim_direction, hidden_dim//2),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim//2, 3),
# #             nn.Sigmoid()  # Normalize colors to [0, 1]
# #         )
        
# #     # def positional_encoding(self, x, L=10):
# #     #     """
# #     #     Apply positional encoding to the input.
        
# #     #     Args:
# #     #         x (torch.Tensor): Input coordinates (position or direction)
# #     #         L (int): Number of encoding functions
            
# #     #     Returns:
# #     #         torch.Tensor: Encoded coordinates with shape [..., 3 + 3*2*L]
# #     #     """
# #     #     encoded = [x]
        
# #     #     for i in range(L):
# #     #         for fn in [torch.sin, torch.cos]:
# #     #             encoded.append(fn(2.0**i * x))
                
# #     #     return torch.cat(encoded, dim=-1)
# #     def positional_encoding(self, x, num_encoding_functions):
# #         if torch.isnan(x).any():
# #             print("❌ NaN detected in input to positional encoding")
# #             raise ValueError("Input to positional encoding contains NaNs")
# #         if torch.isinf(x).any():
# #             print("❌ Inf detected in input to positional encoding")
# #             raise ValueError("Input to positional encoding contains Inf")

# #         encoded = [x]
# #         for i in range(num_encoding_functions):
# #             for fn in [torch.sin, torch.cos]:
# #                 encoded.append(fn(2.0 ** i * x))
# #         return torch.cat(encoded, dim=-1)

# #     def get_rays(self, poses):
# #         """
# #         Generate rays for each pixel in the target image.
        
# #         Args:
# #             poses (torch.Tensor): Camera poses with shape [B, 7] (position + quaternion)
            
# #         Returns:
# #             tuple: (ray_origins, ray_directions) with shapes [B, H*W, 3]
# #         """
# #         B = poses.shape[0]
# #         device = poses.device
        
# #         # Extract positions and convert quaternions to rotation matrices
# #         positions = poses[:, :3]  # [B, 3]
# #         quaternions = poses[:, 3:]  # [B, 4]
        
# #         # Convert quaternions to rotation matrices
# #         R = self.quaternion_to_rotation_matrix(quaternions)  # [B, 3, 3]
        
# #         # Create pixel coordinates grid
# #         i, j = torch.meshgrid(
# #             torch.linspace(0, self.image_width - 1, self.image_width),
# #             torch.linspace(0, self.image_height - 1, self.image_height),
# #             indexing='ij'
# #         )
# #         i, j = i.t().to(device), j.t().to(device)  # Transpose to match image coordinates
        
# #         # Scale to [-1, 1] and create homogeneous coordinates
# #         x = (2 * (i + 0.5) / self.image_width - 1) * torch.tan(torch.tensor(np.pi/4))
# #         y = (2 * (j + 0.5) / self.image_height - 1) * torch.tan(torch.tensor(np.pi/4))
        
# #         # Generate ray directions in camera frame
# #         directions = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)  # [H, W, 3]
# #         directions = directions / torch.norm(directions, dim=-1, keepdim=True)  # Normalize
        
# #         # Repeat for batch
# #         directions = directions.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 3]
        
# #         # Transform ray directions to world frame
# #         directions = directions.reshape(B, -1, 3)  # [B, H*W, 3]
# #         ray_directions = torch.bmm(directions, R)  # [B, H*W, 3]
        
# #         # Set ray origins to camera positions
# #         ray_origins = positions.unsqueeze(1).repeat(1, self.image_height * self.image_width, 1)  # [B, H*W, 3]
        
# #         return ray_origins, ray_directions
    
# #     def quaternion_to_rotation_matrix(self, q):
# #         """
# #         Convert quaternions to rotation matrices.
        
# #         Args:
# #             q (torch.Tensor): Quaternions with shape [..., 4] (qx, qy, qz, qw)
            
# #         Returns:
# #             torch.Tensor: Rotation matrices with shape [..., 3, 3]
# #         """
# #         # Normalize quaternions
# #         q = F.normalize(q, dim=-1)
        
# #         # Extract quaternion components
# #         qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
# #         # Compute rotation matrix elements
# #         R00 = 1 - 2 * (qy**2 + qz**2)
# #         R01 = 2 * (qx * qy - qw * qz)
# #         R02 = 2 * (qx * qz + qw * qy)
        
# #         R10 = 2 * (qx * qy + qw * qz)
# #         R11 = 1 - 2 * (qx**2 + qz**2)
# #         R12 = 2 * (qy * qz - qw * qx)
        
# #         R20 = 2 * (qx * qz - qw * qy)
# #         R21 = 2 * (qy * qz + qw * qx)
# #         R22 = 1 - 2 * (qx**2 + qy**2)
        
# #         # Stack to form rotation matrices
# #         R = torch.stack([
# #             torch.stack([R00, R01, R02], dim=-1),
# #             torch.stack([R10, R11, R12], dim=-1),
# #             torch.stack([R20, R21, R22], dim=-1)
# #         ], dim=-2)
        
# #         return R
    
# #     def render_rays(self, ray_origins, ray_directions, near=2.0, far=6.0, num_samples=64, rand=True):
# #         """
# #         Render rays by sampling points and predicting colors and densities.
        
# #         Args:
# #             ray_origins (torch.Tensor): Ray origin points [B, N, 3]
# #             ray_directions (torch.Tensor): Ray direction vectors [B, N, 3]
# #             near (float): Near plane distance
# #             far (float): Far plane distance
# #             num_samples (int): Number of sample points per ray
# #             rand (bool): Whether to use randomized sampling
            
# #         Returns:
# #             dict: Output containing rendered colors and depths
# #         """
# #         device = ray_origins.device
# #         batch_size, num_rays = ray_origins.shape[:2]
        
# #         # Sample points along each ray
# #         t_vals = torch.linspace(near, far, num_samples, device=device)
        
# #         if rand:
# #             # Add random offset to sample points for better training
# #             mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
# #             upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
# #             lower = torch.cat([t_vals[..., :1], mids], dim=-1)
# #             t_rand = torch.rand(batch_size, num_rays, num_samples, device=device)
# #             t_vals = lower.unsqueeze(0).unsqueeze(0) + (upper - lower).unsqueeze(0).unsqueeze(0) * t_rand
# #         else:
# #             t_vals = t_vals.expand(batch_size, num_rays, num_samples)
        
# #         # Calculate sample point positions along rays
# #         ray_directions_expanded = ray_directions.unsqueeze(2).expand(-1, -1, num_samples, -1)  # [B, N, S, 3]
# #         ray_origins_expanded = ray_origins.unsqueeze(2).expand(-1, -1, num_samples, -1)  # [B, N, S, 3]
# #         sample_points = ray_origins_expanded + t_vals.unsqueeze(-1) * ray_directions_expanded  # [B, N, S, 3]
        
# #         # Reshape for batch processing
# #         flattened_points = sample_points.reshape(-1, 3)  # [B*N*S, 3]
        
# #         # Apply positional encoding
# #         encoded_points = self.positional_encoding(flattened_points, self.num_encoding_functions)  # [B*N*S, 3+3*2*L]
        
# #         # Predict density and features
# #         density_outputs = self.density_net(encoded_points)  # [B*N*S, hidden+1]
# #         sigma = F.relu(density_outputs[..., 0])  # [B*N*S]
# #         features = density_outputs[..., 1:]  # [B*N*S, hidden]
        
# #         # Reshape directions for batch processing
# #         flattened_dirs = ray_directions_expanded.reshape(-1, 3)  # [B*N*S, 3]
        
# #         # Apply positional encoding to directions
# #         encoded_dirs = self.positional_encoding(flattened_dirs, self.num_encoding_functions)  # [B*N*S, 3+3*2*L]
        
# #         # Predict colors using density features and encoded directions
# #         color_input = torch.cat([features, encoded_dirs], dim=-1)  # [B*N*S, hidden+3+3*2*L]
# #         colors = self.color_net(color_input)  # [B*N*S, 3]
        
# #         # Reshape outputs
# #         sigma = sigma.reshape(batch_size, num_rays, num_samples)  # [B, N, S]
# #         colors = colors.reshape(batch_size, num_rays, num_samples, 3)  # [B, N, S, 3]
        
# #         # Perform volume rendering
# #         delta_dist = t_vals[..., 1:] - t_vals[..., :-1]  # [B, N, S-1]
# #         delta_dist = torch.cat([
# #             delta_dist, 
# #             1e10 * torch.ones_like(delta_dist[..., :1])  # Add large value for last sample
# #         ], dim=-1)  # [B, N, S]
        
# #         # Convert sigma to alpha values
# #         alpha = 1.0 - torch.exp(-sigma * delta_dist)  # [B, N, S]
        
# #         # Calculate transmittance
# #         weights = alpha * torch.cumprod(
# #             torch.cat([
# #                 torch.ones_like(alpha[..., :1]), 
# #                 1.0 - alpha[..., :-1]
# #             ], dim=-1),
# #             dim=-1
# #         )  # [B, N, S]
        
# #         # Combine colors with weights
# #         rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=2)  # [B, N, 3]
        
# #         # Calculate depth
# #         depth = torch.sum(weights * t_vals, dim=-1)  # [B, N]
        
# #         return {
# #             'rgb': rgb,
# #             'depth': depth,
# #             'weights': weights
# #         }
    
# #     def forward(self, poses, images, target_pose, num_rays=1024, train_mode=True):
# #         """
# #         Forward pass: use input poses and images to render a new view from target_pose.
        
# #         Args:
# #             poses (torch.Tensor): Input camera poses [B, N, 7]
# #             images (torch.Tensor): Input images [B, N, H, W, 3]
# #             target_pose (torch.Tensor): Target camera pose [B, 7]
# #             num_rays (int): Number of rays to sample during training
# #             train_mode (bool): Whether in training mode (use ray sampling)
            
# #         Returns:
# #             dict: Output containing the rendered image and depth
# #         """
# #         B = poses.shape[0]
# #         device = poses.device
        
# #         # Generate rays for target view
# #         ray_origins, ray_directions = self.get_rays(target_pose)
        
# #         # if train_mode and num_rays < self.image_height * self.image_width:
# #         #     # Sample a subset of rays during training
# #         #     ray_indices = torch.randint(
# #         #         0, self.image_height * self.image_width, 
# #         #         (B, num_rays), 
# #         #         device=device
# #         #     )
# #         if train_mode and num_rays < self.image_height * self.image_width:
# #     # Sample a subset of rays per batch item
# #             ray_indices = torch.randint(
# #                 0, self.image_height * self.image_width,
# #                 (B, num_rays),
# #                 device=device
# #             )  # [B, num_rays]
            
# #             ray_origins = torch.gather(
# #                 ray_origins, 1, ray_indices.unsqueeze(-1).expand(-1, -1, 3)
# #             )  # [B, num_rays, 3]

# #             ray_directions = torch.gather(
# #                 ray_directions, 1, ray_indices.unsqueeze(-1).expand(-1, -1, 3)
# #     )  # [B, num_rays, 3]

            
# #             # Compute batched indices into [B * H * W]
# #             # ray_stride = self.image_height * self.image_width
# #             # batched_indices = ray_indices + torch.arange(B, device=device).unsqueeze(1) * ray_stride

# #             # # Debug: check for out-of-bounds
# #             # if batched_indices.max() >= ray_origins.reshape(-1, 3).shape[0]:
# #             #     print("❌ Out-of-bounds batched_indices")
# #             #     print("🧾 ray_indices.max():", ray_indices.max().item())
# #             #     print("📏 ray_stride:", ray_stride)
# #             #     print("📏 ray_origins total:", ray_origins.reshape(-1, 3).shape[0])
# #             #     print("📏 batched_indices.max():", batched_indices.max().item())
# #             #     raise IndexError("Index out of bounds in batched_indices computation")

# #             # flat_ray_origins = ray_origins.reshape(-1, 3)
# #             # if batched_indices.max() >= flat_ray_origins.shape[0]:
# #             #     print("❌ Index out of bounds in ray_origins")
# #             #     print("🔍 batched_indices.max():", batched_indices.max().item())
# #             #     print("🟢 ray_origins.shape:", ray_origins.shape)
# #             #     print("🟢 flat_ray_origins.shape[0]:", flat_ray_origins.shape[0])
# #             #     raise IndexError("Index out of bounds when accessing ray_origins")

# #             # ray_origins = flat_ray_origins[batched_indices].reshape(B, num_rays, 3)

# #             # ray_directions = ray_directions.reshape(-1, 3)[batched_indices].reshape(B, num_rays, 3)
# #                     # Generate rays for target view
# #         ray_origins, ray_directions = self.get_rays(target_pose)

# #         if train_mode and num_rays < self.image_height * self.image_width:
# #             # Sample a subset of rays per batch item
# #             ray_indices = torch.randint(
# #                 0, self.image_height * self.image_width,
# #                 (B, num_rays),
# #                 device=device
# #             )  # [B, num_rays]

# #             # Safe gather on 3D ray origins and directions
# #             ray_origins = torch.gather(
# #                 ray_origins, 1, ray_indices.unsqueeze(-1).expand(-1, -1, 3)
# #             )  # [B, num_rays, 3]

# #             ray_directions = torch.gather(
# #                 ray_directions, 1, ray_indices.unsqueeze(-1).expand(-1, -1, 3)
# #             )  # [B, num_rays, 3]

        
# #         # Render rays
# #         render_output = self.render_rays(
# #             ray_origins,
# #             ray_directions,
# #             near=2.0,
# #             far=6.0,
# #             num_samples=64,
# #             rand=train_mode
# #         )
# #         if train_mode:
# #             # Compute training loss (MSE between rendered RGB and target)
# #             target_pixels = images[:, -1]  # Assume last image in batch is the target
# #             target_pixels = target_pixels.reshape(B, -1, 3)  # [B, H*W, 3]
# #             target_pixels = torch.gather(
# #                 target_pixels, 1, ray_indices.unsqueeze(-1).expand(-1, -1, 3)
# #             )  # [B, num_rays, 3]

# #             # Mean squared error between rendered and ground truth
# #             loss = F.mse_loss(render_output["rgb"], target_pixels)

# #             render_output["loss"] = loss

# #         return render_output

        
# #         # # Reshape rendered image to HxW format
# #         # if not train_mode:
# #         #     render_output['rgb'] = render_output['rgb'].reshape(B, self.image_height, self.image_width, 3)
# #         #     render_output['depth'] = render_output['depth'].reshape(B, self.image_height, self.image_width)
        
# #         # return render_output



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# print("Loading TinyNeRF version with target_image support")
# class TinyNeRF(nn.Module):
#     def __init__(self,
#                  num_encoding_functions=10,
#                  hidden_dim=128,
#                  num_layers=4,
#                  image_height=256,
#                  image_width=256):
#         super().__init__()
#         self.num_encoding_functions = num_encoding_functions
#         self.in_dim_position = 3 + 3 * 2 * num_encoding_functions
#         self.in_dim_direction = 3 + 3 * 2 * num_encoding_functions
#         self.image_height = image_height
#         self.image_width = image_width

#         # Density network
#         density_layers = [nn.Linear(self.in_dim_position, hidden_dim), nn.ReLU()]
#         for _ in range(num_layers - 2):
#             density_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
#         density_layers.append(nn.Linear(hidden_dim, hidden_dim + 1))
#         self.density_net = nn.Sequential(*density_layers)

#         # Color network
#         self.color_net = nn.Sequential(
#             nn.Linear(hidden_dim + self.in_dim_direction, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 3),
#             nn.Sigmoid()
#         )

#     def positional_encoding(self, x, L):
#         encoded = [x]
#         for i in range(L):
#             for fn in [torch.sin, torch.cos]:
#                 encoded.append(fn(2.0 ** i * x))
#         return torch.cat(encoded, dim=-1)

#     def quaternion_to_rotation_matrix(self, q):
#         q = F.normalize(q, dim=-1)
#         qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
#         R = torch.stack([
#             torch.stack([1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy], dim=-1),
#             torch.stack([2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx], dim=-1),
#             torch.stack([2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2], dim=-1)
#         ], dim=-2)
#         return R

#     def get_rays(self, poses):
#         B = poses.shape[0]
#         device = poses.device
#         positions = poses[:, :3]
#         quaternions = poses[:, 3:]
#         R = self.quaternion_to_rotation_matrix(quaternions)

#         i, j = torch.meshgrid(
#             torch.linspace(0, self.image_width - 1, self.image_width),
#             torch.linspace(0, self.image_height - 1, self.image_height),
#             indexing='ij'
#         )
#         i, j = i.t().to(device), j.t().to(device)
#         x = (2 * (i + 0.5) / self.image_width - 1) * torch.tan(torch.tensor(np.pi/4))
#         y = (2 * (j + 0.5) / self.image_height - 1) * torch.tan(torch.tensor(np.pi/4))
#         directions = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)
#         directions = directions / torch.norm(directions, dim=-1, keepdim=True)
#         directions = directions.unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, -1, 3)
#         ray_directions = torch.bmm(directions, R)
#         ray_origins = positions.unsqueeze(1).repeat(1, self.image_height * self.image_width, 1)
#         return ray_origins, ray_directions

#     def render_rays(self, ray_origins, ray_directions, near=2.0, far=6.0, num_samples=64, rand=True):
#         device = ray_origins.device
#         B, N, _ = ray_origins.shape
#         t_vals = torch.linspace(near, far, num_samples, device=device)
#         if rand:
#             mids = 0.5 * (t_vals[1:] + t_vals[:-1])
#             upper = torch.cat([mids, t_vals[-1:]], dim=0)
#             lower = torch.cat([t_vals[:1], mids], dim=0)
#             t_rand = torch.rand(B, N, num_samples, device=device)
#             t_vals = lower[None, None, :] + (upper - lower)[None, None, :] * t_rand
#         else:
#             t_vals = t_vals.expand(B, N, num_samples)

#         pts = ray_origins[..., None, :] + ray_directions[..., None, :] * t_vals[..., :, None]
#         pts_flat = pts.view(-1, 3)
#         dirs_flat = ray_directions[..., None, :].expand(B, N, num_samples, 3).contiguous().view(-1, 3)

#         encoded_pts = self.positional_encoding(pts_flat, self.num_encoding_functions)
#         encoded_dirs = self.positional_encoding(dirs_flat, self.num_encoding_functions)

#         density_out = self.density_net(encoded_pts)
#         sigma = F.relu(density_out[..., 0])
#         features = density_out[..., 1:]

#         color_input = torch.cat([features, encoded_dirs], dim=-1)
#         colors = self.color_net(color_input)

#         sigma = sigma.view(B, N, num_samples)
#         colors = colors.view(B, N, num_samples, 3)

#         delta = t_vals[..., 1:] - t_vals[..., :-1]
#         delta = torch.cat([delta, 1e10 * torch.ones_like(delta[..., :1])], dim=-1)
#         alpha = 1.0 - torch.exp(-sigma * delta)
#         weights = alpha * torch.cumprod(
#             torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]], dim=-1),
#             dim=-1
#         )
#         rgb = torch.sum(weights[..., None] * colors, dim=-2)
#         depth = torch.sum(weights * t_vals, dim=-1)
#         return {'rgb': rgb, 'depth': depth, 'weights': weights}

#     def forward(self, poses, images, target_pose, target_image=None, num_rays=1024, train_mode=True):
#         B = poses.shape[0]
#         device = poses.device
#         ray_origins, ray_directions = self.get_rays(target_pose)
#         if train_mode and num_rays < self.image_height * self.image_width:
#             ray_indices = torch.randint(0, self.image_height * self.image_width, (B, num_rays), device=device)
#             ray_origins = torch.gather(ray_origins, 1, ray_indices.unsqueeze(-1).expand(-1, -1, 3))
#             ray_directions = torch.gather(ray_directions, 1, ray_indices.unsqueeze(-1).expand(-1, -1, 3))
#         else:
#             ray_indices = None

#         out = self.render_rays(ray_origins, ray_directions, rand=train_mode)

#         if train_mode and ray_indices is not None:
#             target_pixels = images[:, -1]
#             target_pixels = target_pixels.view(B, -1, 3)
#             target_pixels = torch.gather(target_pixels, 1, ray_indices.unsqueeze(-1).expand(-1, -1, 3))
#             out['loss'] = F.mse_loss(out['rgb'], target_pixels)

#         return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10

class TinyNeRF(nn.Module):
    """
    A lightweight, differentiable Neural Radiance Field (NeRF) implementation.
    
    This implementation focuses on:
    1. Differentiability (for end-to-end training)
    2. Efficiency (for faster training)
    3. Simplicity (fewer parameters than full NeRF)
    
    The model renders novel views given input images and camera poses.
    """
    def __init__(self,
                 num_encoding_functions=10,
                 hidden_dim=128,
                 num_layers=4,
                 image_height=256,
                 image_width=256):
        super().__init__()
        
        # Positional encoding dimensions
        self.num_encoding_functions = num_encoding_functions
        self.in_dim_position = 3 + 3 * 2 * num_encoding_functions  # (x,y,z) + positional encoding
        self.in_dim_direction = 3 + 3 * 2 * num_encoding_functions  # (d_x,d_y,d_z) + positional encoding
        
        # Image dimensions
        self.image_height = image_height
        self.image_width = image_width
        
        # MLP for density prediction
        density_layers = []
        density_layers.append(nn.Linear(self.in_dim_position, hidden_dim))
        density_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            density_layers.append(nn.Linear(hidden_dim, hidden_dim))
            density_layers.append(nn.ReLU())
            
        density_layers.append(nn.Linear(hidden_dim, hidden_dim + 1))  # +1 for density sigma
        
        self.density_net = nn.Sequential(*density_layers)
        
        # MLP for color prediction
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + self.in_dim_direction, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 3),
            nn.Sigmoid()  # Normalize colors to [0, 1]
        )
        
    def positional_encoding(self, x, L=10):
        """
        Apply positional encoding to the input.
        
        Args:
            x (torch.Tensor): Input coordinates (position or direction)
            L (int): Number of encoding functions
            
        Returns:
            torch.Tensor: Encoded coordinates with shape [..., 3 + 3*2*L]
        """
        encoded = [x]
        
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                encoded.append(fn(2.0**i * x))
                
        return torch.cat(encoded, dim=-1)
    
    def get_rays(self, poses):
        """
        Generate rays for each pixel in the target image.
        
        Args:
            poses (torch.Tensor): Camera poses with shape [B, 7] (position + quaternion)
            
        Returns:
            tuple: (ray_origins, ray_directions) with shapes [B, H*W, 3]
        """
        B = poses.shape[0]
        device = poses.device
        
        # Extract positions and convert quaternions to rotation matrices
        positions = poses[:, :3]  # [B, 3]
        quaternions = poses[:, 3:]  # [B, 4]
        
        # Convert quaternions to rotation matrices
        R = self.quaternion_to_rotation_matrix(quaternions)  # [B, 3, 3]
        
        # Create pixel coordinates grid
        i, j = torch.meshgrid(
            torch.linspace(0, self.image_width - 1, self.image_width),
            torch.linspace(0, self.image_height - 1, self.image_height),
            indexing='ij'
        )
        i, j = i.t().to(device), j.t().to(device)  # Transpose to match image coordinates
        
        # Scale to [-1, 1] and create homogeneous coordinates
        x = (2 * (i + 0.5) / self.image_width - 1) * torch.tan(torch.tensor(np.pi/4))
        y = (2 * (j + 0.5) / self.image_height - 1) * torch.tan(torch.tensor(np.pi/4))
        
        # Generate ray directions in camera frame
        directions = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)  # [H, W, 3]
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)  # Normalize
        
        # Repeat for batch
        directions = directions.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 3]
        
        # Transform ray directions to world frame
        directions = directions.reshape(B, -1, 3)  # [B, H*W, 3]
        ray_directions = torch.bmm(directions, R)  # [B, H*W, 3]
        
        # Set ray origins to camera positions
        ray_origins = positions.unsqueeze(1).repeat(1, self.image_height * self.image_width, 1)  # [B, H*W, 3]
        
        return ray_origins, ray_directions
    
    def quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternions to rotation matrices.
        
        Args:
            q (torch.Tensor): Quaternions with shape [..., 4] (qx, qy, qz, qw)
            
        Returns:
            torch.Tensor: Rotation matrices with shape [..., 3, 3]
        """
        # Normalize quaternions
        q = F.normalize(q, dim=-1)
        
        # Extract quaternion components
        qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # Compute rotation matrix elements
        R00 = 1 - 2 * (qy**2 + qz**2)
        R01 = 2 * (qx * qy - qw * qz)
        R02 = 2 * (qx * qz + qw * qy)
        
        R10 = 2 * (qx * qy + qw * qz)
        R11 = 1 - 2 * (qx**2 + qz**2)
        R12 = 2 * (qy * qz - qw * qx)
        
        R20 = 2 * (qx * qz - qw * qy)
        R21 = 2 * (qy * qz + qw * qx)
        R22 = 1 - 2 * (qx**2 + qy**2)
        
        # Stack to form rotation matrices
        R = torch.stack([
            torch.stack([R00, R01, R02], dim=-1),
            torch.stack([R10, R11, R12], dim=-1),
            torch.stack([R20, R21, R22], dim=-1)
        ], dim=-2)
        
        return R
    
    def render_rays(self, ray_origins, ray_directions, near=2.0, far=6.0, num_samples=64, rand=True):
        """
        Render rays by sampling points and predicting colors and densities.
        
        Args:
            ray_origins (torch.Tensor): Ray origin points [B, N, 3]
            ray_directions (torch.Tensor): Ray direction vectors [B, N, 3]
            near (float): Near plane distance
            far (float): Far plane distance
            num_samples (int): Number of sample points per ray
            rand (bool): Whether to use randomized sampling
            
        Returns:
            dict: Output containing rendered colors and depths
        """
        device = ray_origins.device
        batch_size, num_rays = ray_origins.shape[:2]
        
        # Sample points along each ray
        t_vals = torch.linspace(near, far, num_samples, device=device)
        
        if rand:
            # Add random offset to sample points for better training
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(batch_size, num_rays, num_samples, device=device)
            t_vals = lower.unsqueeze(0).unsqueeze(0) + (upper - lower).unsqueeze(0).unsqueeze(0) * t_rand
        else:
            t_vals = t_vals.expand(batch_size, num_rays, num_samples)
        
        # Calculate sample point positions along rays
        ray_directions_expanded = ray_directions.unsqueeze(2).expand(-1, -1, num_samples, -1)  # [B, N, S, 3]
        ray_origins_expanded = ray_origins.unsqueeze(2).expand(-1, -1, num_samples, -1)  # [B, N, S, 3]
        sample_points = ray_origins_expanded + t_vals.unsqueeze(-1) * ray_directions_expanded  # [B, N, S, 3]
        
        # Reshape for batch processing
        flattened_points = sample_points.reshape(-1, 3)  # [B*N*S, 3]
        
        # Apply positional encoding
        encoded_points = self.positional_encoding(flattened_points, self.num_encoding_functions)  # [B*N*S, 3+3*2*L]
        
        # Predict density and features
        density_outputs = self.density_net(encoded_points)  # [B*N*S, hidden+1]
        sigma = F.relu(density_outputs[..., 0])  # [B*N*S]
        features = density_outputs[..., 1:]  # [B*N*S, hidden]
        
        # Reshape directions for batch processing
        flattened_dirs = ray_directions_expanded.reshape(-1, 3)  # [B*N*S, 3]
        
        # Apply positional encoding to directions
        encoded_dirs = self.positional_encoding(flattened_dirs, self.num_encoding_functions)  # [B*N*S, 3+3*2*L]
        
        # Predict colors using density features and encoded directions
        color_input = torch.cat([features, encoded_dirs], dim=-1)  # [B*N*S, hidden+3+3*2*L]
        colors = self.color_net(color_input)  # [B*N*S, 3]
        
        # Reshape outputs
        sigma = sigma.reshape(batch_size, num_rays, num_samples)  # [B, N, S]
        colors = colors.reshape(batch_size, num_rays, num_samples, 3)  # [B, N, S, 3]
        
        # Perform volume rendering
        delta_dist = t_vals[..., 1:] - t_vals[..., :-1]  # [B, N, S-1]
        delta_dist = torch.cat([
            delta_dist, 
            1e10 * torch.ones_like(delta_dist[..., :1])  # Add large value for last sample
        ], dim=-1)  # [B, N, S]
        
        # Convert sigma to alpha values
        alpha = 1.0 - torch.exp(-sigma * delta_dist)  # [B, N, S]
        
        # Calculate transmittance
        weights = alpha * torch.cumprod(
            torch.cat([
                torch.ones_like(alpha[..., :1]), 
                1.0 - alpha[..., :-1]
            ], dim=-1),
            dim=-1
        )  # [B, N, S]
        
        # Combine colors with weights
        rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=2)  # [B, N, 3]
        
        # Calculate depth
        depth = torch.sum(weights * t_vals, dim=-1)  # [B, N]
        
        return {
            'rgb': rgb,
            'depth': depth,
            'weights': weights
        }
    
    def forward(self,
                poses,
                images,
                target_pose,
                target_image=None,
                num_rays=None,
                train=True):
        """
        Forward pass: use input poses and images to render a new view from target_pose.
        
        Args:
            poses (torch.Tensor): Input camera poses [B, N, 7]
            images (torch.Tensor): Input images [B, N, H, W, 3]
            target_pose (torch.Tensor): Target camera pose [B, 7]
            target_image (torch.Tensor, optional): Target image for loss calculation
            num_rays (int, optional): Number of rays to sample during training
            train (bool): Whether in training mode (use ray sampling)
            
        Returns:
            dict: Output containing the rendered image, depth, and loss if target_image is provided
        """
        B = poses.shape[0]
        device = poses.device
        H, W = self.image_height, self.image_width
        
        # Generate rays for target view
        ray_origins, ray_directions = self.get_rays(target_pose)
        
        # if training and we want to subsample fewer than H*W rays:
        if train and num_rays is not None and num_rays < H * W:
            # flatten H×W → N = H*W
            ray_origins = ray_origins.view(B, H*W, 3)
            ray_directions = ray_directions.view(B, H*W, 3)

            # sample indices
            idx = torch.randint(0, H*W, (B, num_rays), device=device)  # [B, num_rays]

            # gather
            ray_origins   = torch.gather(ray_origins,   1, idx.unsqueeze(-1).expand(-1, -1, 3))
            ray_directions= torch.gather(ray_directions,1, idx.unsqueeze(-1).expand(-1, -1, 3))

            # remember them so we can index target pixels later
            sel_indices = idx
        else:
            # no subsampling → use all rays
            ray_origins    = ray_origins.view(B, H*W, 3)
            ray_directions = ray_directions.view(B, H*W, 3)
            sel_indices    = torch.arange(H*W, device=device).unsqueeze(0).expand(B, -1)
        
        # Render rays
        render_output = self.render_rays(
            ray_origins,
            ray_directions,
            near=2.0,
            far=6.0,
            num_samples=64,
            rand=train
        )
        
        rendered_rgb = render_output['rgb']
        
        outputs = {"rgb": rendered_rgb}

        # compute loss & PSNR if we have a target
        if train and target_image is not None:
            # flatten target_image [B, C, H, W] → [B, H*W, C]
            B_,C,H_,W_ = target_image.shape[0], target_image.shape[1], H, W
            tgt = target_image.view(B_, C, H_*W_).permute(0,2,1)  # [B, H*W, C]

            # pick only the same rays we rendered
            sel_tgt = torch.gather(
                tgt, 1, sel_indices.unsqueeze(-1).expand(-1, -1, C)
            ).reshape(-1, C)  # [B*num_rays, C]

            # compute MSE loss
            loss = F.mse_loss(rendered_rgb.reshape(-1, 3), sel_tgt)
            outputs["loss"] = loss

            # PSNR = –10·log₁₀(MSE)
            psnr = -10 * log10(loss.item() if isinstance(loss, torch.Tensor) else loss)
            outputs["psnr"] = torch.tensor(psnr, device=rendered_rgb.device)
        
        # Reshape rendered image to HxW format if not in training mode
        if not train:
            outputs["rgb"] = outputs["rgb"].reshape(B, H, W, 3)
            if "depth" in render_output:
                outputs["depth"] = render_output["depth"].reshape(B, H, W)
        
        return outputs
