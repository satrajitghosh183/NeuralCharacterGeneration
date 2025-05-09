# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .tiny_nerf import TinyNeRF


# class WeightedTinyNeRF(nn.Module):
#     """
#     Extension of TinyNeRF that supports weighted view selection.
    
#     This model:
#     1. Takes predicted selection weights from the transformer
#     2. Applies soft view selection (via softmax)
#     3. Feeds selected views into TinyNeRF
    
#     Used for end-to-end training where view selection influences reconstruction quality.
#     """
#     def __init__(self,
#                  num_encoding_functions=10,
#                  hidden_dim=128,
#                  num_layers=4,
#                  image_height=256,
#                  image_width=256,
#                  top_k=5):
#         super().__init__()
        
#         # Base NeRF model
#         self.nerf = TinyNeRF(
#             num_encoding_functions=num_encoding_functions,
#             hidden_dim=hidden_dim,
#             num_layers=num_layers,
#             image_height=image_height,
#             image_width=image_width
#         )
        
#         # Number of views to select
#         self.top_k = top_k
    
#     def get_top_k_views(self, selection_weights, images, poses, k=None):
#         """
#         Select the top-k views based on selection weights.
        
#         Args:
#             selection_weights (torch.Tensor): View selection scores [B, N]
#             images (torch.Tensor): Input images [B, N, H, W, 3]
#             poses (torch.Tensor): Input poses [B, N, 7]
#             k (int, optional): Number of views to select (defaults to self.top_k)
            
#         Returns:
#             tuple: (selected_images, selected_poses, selection_probs)
#                 - selected_images: Top-k selected images [B, k, H, W, 3]
#                 - selected_poses: Top-k selected poses [B, k, 7]
#                 - selection_probs: Selection probabilities [B, N]
#         """
#         if k is None:
#             k = self.top_k
            
#         B, N = selection_weights.shape
        
#         # Convert to probabilities
#         selection_probs = F.softmax(selection_weights, dim=-1)  # [B, N]
        
#         if self.training:
#             # During training: Use soft selection (weighted sum)
#             # This allows gradients to flow back to the transformer
            
#             # Top-k hard selection for memory efficiency
#             top_k_values, top_k_indices = torch.topk(selection_probs, k=k, dim=1)  # [B, k]
            
#             # Normalize the top-k probabilities
#             top_k_probs = top_k_values / top_k_values.sum(dim=1, keepdim=True)  # [B, k]
            
#             # Gather top-k images and poses
#             batch_indices = torch.arange(B, device=selection_weights.device).unsqueeze(1).expand(-1, k)
            
#             selected_images = images[batch_indices, top_k_indices]  # [B, k, H, W, 3]
#             selected_poses = poses[batch_indices, top_k_indices]  # [B, k, 7]
            
#             # Return with normalized probabilities for the top-k
#             return selected_images, selected_poses, top_k_probs
#         else:
#             # During inference: Use hard selection (just take top-k views)
#             _, top_k_indices = torch.topk(selection_probs, k=k, dim=1)  # [B, k]
            
#             # Create batch indices for gather operation
#             batch_indices = torch.arange(B, device=selection_weights.device).unsqueeze(1).expand(-1, k)
            
#             # Gather top-k images and poses
#             selected_images = images[batch_indices, top_k_indices]  # [B, k, H, W, 3]
#             selected_poses = poses[batch_indices, top_k_indices]  # [B, k, 7]
            
#             # For inference, we create one-hot vectors for selection probabilities
#             one_hot_probs = torch.zeros_like(selection_probs)
#             one_hot_probs.scatter_(1, top_k_indices, 1.0 / k)  # Equal weights for selected views
            
#             return selected_images, selected_poses, one_hot_probs
    
#     def forward(self, selection_weights, images, poses, target_pose, target_image=None, num_rays=1024):
#         """
#         Forward pass through the weighted TinyNeRF model.
        
#         Args:
#             selection_weights (torch.Tensor): View selection scores [B, N]
#             images (torch.Tensor): Input images [B, N, H, W, 3]
#             poses (torch.Tensor): Input poses [B, N, 7]
#             target_pose (torch.Tensor): Target camera pose [B, 7]
#             target_image (torch.Tensor, optional): Target image for loss calculation [B, H, W, 3]
#             num_rays (int): Number of rays to sample during training
            
#         Returns:
#             dict: Output containing:
#                 - rendered image and depth
#                 - selection probabilities
#                 - loss (if target_image is provided)
#         """
#         # Select top-k views based on selection weights
#         selected_images, selected_poses, selection_probs = self.get_top_k_views(
#             selection_weights, images, poses
#         )
        
#         # Determine train mode from target_image (None in inference)
#         train_mode = target_image is not None
        
#         render_output = self.nerf(
#         selected_poses, 
#         selected_images, 
#         target_pose, 
#         num_rays=num_rays,
#         train_mode=train_mode
#     )

#         if render_output is None:
#             raise RuntimeError("TinyNeRF forward pass returned None. Check for issues in selected_poses, selected_images, or target_pose.")

#         # Add selection probabilities to output
#         # render_output['selection_probs'] = selection_probs
        
#         # # Calculate loss if target image is provided
#         # if target_image is not None:
#         #     # Sample pixels from target image to match rendered rays
#         #     B, H, W, _ = target_image.shape
            
#         #     if num_rays < H * W:
#         #         # Re-create the same random rays as in the NeRF forward pass
#         #         ray_indices = torch.randint(0, H * W, (B, num_rays), device=target_image.device)
#         #         batched_indices = ray_indices + torch.arange(B, device=target_image.device).unsqueeze(1) * (H * W)
                
#         #         target_pixels = target_image.reshape(B * H * W, 3)[batched_indices].reshape(B, num_rays, 3)
#         #     else:
#         #         target_pixels = target_image.reshape(B, H * W, 3)
            
#         #     # Calculate MSE loss between rendered RGB and target RGB
#         #     mse_loss = F.mse_loss(render_output['rgb'], target_pixels)
            
#         #     # Optional: PSNR metric (useful for evaluation)
#         #     psnr = -10.0 * torch.log10(mse_loss)
            
#         #     render_output['loss'] = mse_loss
#         #     render_output['psnr'] = psnr
        
#         # return render_output
#         # Add selection probabilities to output
#         render_output['selection_probs'] = selection_probs

#         # Calculate loss if target image is provided
#         if target_image is not None:
#             # Ensure target_image is in NHWC format: (B, H, W, 3)
#             if target_image.shape[1] == 3 and target_image.shape[-1] != 3:
#                 # Convert from NCHW to NHWC
#                 target_image = target_image.permute(0, 2, 3, 1).contiguous()

#             # Validate expected shape
#             B, H, W, C = target_image.shape
#             assert C == 3, f"Expected 3-channel RGB image, got shape: {target_image.shape}"

#             # Sample pixels from target image to match rendered rays
#             if num_rays < H * W:
#                 # Re-create same random ray indices
#                 ray_indices = torch.randint(0, H * W, (B, num_rays), device=target_image.device)
#                 batched_indices = ray_indices + torch.arange(B, device=target_image.device).unsqueeze(1) * (H * W)

#                 # Flatten and gather sampled pixels
#                 target_pixels = target_image.view(B * H * W, 3)[batched_indices].view(B, num_rays, 3)
#             else:
#                 # Use all pixels
#                 target_pixels = target_image.view(B, H * W, 3)

#             # Calculate MSE loss between NeRF output and target image
#             mse_loss = F.mse_loss(render_output['rgb'], target_pixels)

#             # Compute PSNR
#             psnr = -10.0 * torch.log10(mse_loss + 1e-8)  # avoid log(0)

#             render_output['loss'] = mse_loss
#             render_output['psnr'] = psnr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10
from .tiny_nerf import TinyNeRF


class WeightedTinyNeRF(nn.Module):
    """
    Extension of TinyNeRF that supports weighted view selection.
    
    This model:
    1. Takes predicted selection weights from the transformer
    2. Applies soft view selection (via softmax)
    3. Feeds selected views into TinyNeRF
    
    Used for end-to-end training where view selection influences reconstruction quality.
    """
    def __init__(self,
                 num_encoding_functions=10,
                 hidden_dim=128,
                 num_layers=4,
                 image_height=256,
                 image_width=256,
                 top_k=5):
        super().__init__()
        
        # Base NeRF model
        self.nerf = TinyNeRF(
            num_encoding_functions=num_encoding_functions,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            image_height=image_height,
            image_width=image_width
        )
        
        # Number of views to select
        self.top_k = top_k
        self.image_height = image_height
        self.image_width = image_width
    
    def get_top_k_views(self, selection_weights, images, poses, k=None):
        """
        Select the top-k views based on selection weights.
        
        Args:
            selection_weights (torch.Tensor): View selection scores [B, N]
            images (torch.Tensor): Input images [B, N, H, W, 3]
            poses (torch.Tensor): Input poses [B, N, 7]
            k (int, optional): Number of views to select (defaults to self.top_k)
            
        Returns:
            tuple: (selected_images, selected_poses, selection_probs)
                - selected_images: Top-k selected images [B, k, H, W, 3]
                - selected_poses: Top-k selected poses [B, k, 7]
                - selection_probs: Selection probabilities [B, N]
        """
        if k is None:
            k = self.top_k
            
        B, N = selection_weights.shape
        
        # Convert to probabilities
        selection_probs = F.softmax(selection_weights, dim=-1)  # [B, N]
        
        if self.training:
            # During training: Use soft selection (weighted sum)
            # This allows gradients to flow back to the transformer
            
            # Top-k hard selection for memory efficiency
            top_k_values, top_k_indices = torch.topk(selection_probs, k=k, dim=1)  # [B, k]
            
            # Normalize the top-k probabilities
            top_k_probs = top_k_values / top_k_values.sum(dim=1, keepdim=True)  # [B, k]
            
            # Gather top-k images and poses
            batch_indices = torch.arange(B, device=selection_weights.device).unsqueeze(1).expand(-1, k)
            
            selected_images = images[batch_indices, top_k_indices]  # [B, k, H, W, 3]
            selected_poses = poses[batch_indices, top_k_indices]  # [B, k, 7]
            
            # Return with normalized probabilities for the top-k
            return selected_images, selected_poses, top_k_probs
        else:
            # During inference: Use hard selection (just take top-k views)
            _, top_k_indices = torch.topk(selection_probs, k=k, dim=1)  # [B, k]
            
            # Create batch indices for gather operation
            batch_indices = torch.arange(B, device=selection_weights.device).unsqueeze(1).expand(-1, k)
            
            # Gather top-k images and poses
            selected_images = images[batch_indices, top_k_indices]  # [B, k, H, W, 3]
            selected_poses = poses[batch_indices, top_k_indices]  # [B, k, 7]
            
            # For inference, we create one-hot vectors for selection probabilities
            one_hot_probs = torch.zeros_like(selection_probs)
            one_hot_probs.scatter_(1, top_k_indices, 1.0 / k)  # Equal weights for selected views
            
            return selected_images, selected_poses, one_hot_probs
    
    def forward(self,
                selection_weights,
                images,
                poses,
                target_pose,
                target_image=None,
                num_rays=None,
                train=True):
        """
        Forward pass through the weighted TinyNeRF model.
        
        Args:
            selection_weights (torch.Tensor): View selection scores [B, N]
            images (torch.Tensor): Input images [B, N, H, W, 3]
            poses (torch.Tensor): Input poses [B, N, 7]
            target_pose (torch.Tensor): Target camera pose [B, 7]
            target_image (torch.Tensor, optional): Target image for loss calculation
            num_rays (int, optional): Number of rays to sample during training
            train (bool): Whether in training mode
            
        Returns:
            dict: Output containing rendered image, depth, selection probabilities, and loss if target_image is provided
        """
        # Select top-k views based on selection weights
        selected_images, selected_poses, selection_probs = self.get_top_k_views(
            selection_weights, images, poses
        )
        
        # Determine train mode (True in training, False in inference)
        train_mode = train
        
        # Render target view using selected views
        nerf_output = self.nerf(
        poses=selected_poses, 
        images=selected_images, 
        target_pose=target_pose, 
        target_image=target_image,
        num_rays=num_rays,
        train=train_mode  # Change 'train' to 'train_mode'
    )
            
        # Add selection probabilities to output
        nerf_output['selection_probs'] = selection_probs
        
        return nerf_output