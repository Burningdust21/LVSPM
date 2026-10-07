from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.types import LVSPM
from .loss import Loss
from torchvision.models import vgg19
import torch.nn as nn
import scipy.io
import urllib.request
from pathlib import Path

@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    use_perceptual: bool


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg

# The perception loss code is modified from class class https://github.com/Haian-Jin/LVSM/blob/ebeff4989a3e1ec38fcd51ae24919d0eadf38c8f/model/loss.py#L16
class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()

    def _build_vgg(self):
        """Create VGG model with average pooling instead of max pooling."""
        model = vgg19()
        # Replace max pooling with average pooling
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)

        return model.to(self.device).eval()

    def _load_weights(self):
        """Load pre-trained VGG weights. """
        weight_file = Path("./checkpoints/imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)

        if not weight_file.exists():
            url = "https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
            urllib.request.urlretrieve(url, weight_file)

        # Load MatConvNet weights
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]

        # Layer indices and filter sizes
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]

        # Transfer weights to PyTorch model
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # Set weights
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)

                # Set biases
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)

    def _setup_feature_blocks(self):
        """Create feature extraction blocks at different network depths."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()

        # Create sequential blocks
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i+1]]))
            self.blocks.append(block.to(self.device).eval())

        # Freeze all parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

    def _extract_features(self, x):
        """Extract features from each block."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def _preprocess_images(self, images):
        """Convert images to VGG input format."""
        # VGG mean values for ImageNet
        mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(images.device)
        return images * 255.0 - mean

    @staticmethod
    def _compute_error(real, fake):
        return torch.mean(torch.abs(real - fake))
    @torch.compile(fullgraph=True, dynamic=False)
    def forward(self, pred_img, target_img, normalize=False):
        # if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
        #     pred_img = 2.0 * pred_img  - 1.0
        #     target_img = 2.0 * target_img  - 1.0
        """Compute perceptual loss between prediction and target."""
        # Preprocess images
        target_img_p = self._preprocess_images(target_img)
        pred_img_p = self._preprocess_images(pred_img)

        # Extract features
        target_features = self._extract_features(target_img_p)
        pred_features = self._extract_features(pred_img_p)

        # Pixel-level error
        e0 = self._compute_error(target_img_p, pred_img_p)

        # Feature-level errors with scaling factors
        e1 = self._compute_error(target_features[0], pred_features[0]) / 2.6
        e2 = self._compute_error(target_features[1], pred_features[1]) / 4.8
        e3 = self._compute_error(target_features[2], pred_features[2]) / 3.7
        e4 = self._compute_error(target_features[3], pred_features[3]) / 5.6
        e5 = self._compute_error(target_features[4], pred_features[4]) * 10 / 1.5

        # Combine all errors and normalize
        total_loss = (e0 + e1 + e2 + e3 + e4 + e5) / 255.0

        return total_loss

class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS | PerceptualLoss

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)
        if cfg.lpips.use_perceptual:
            self.lpips = PerceptualLoss().eval()
        else:
            self.lpips = LPIPS(net="vgg").eval()
        self.lpips = torch.compile(self.lpips)
        convert_to_buffer(self.lpips, persistent=False)
        for param in  self.lpips.parameters():
            param.requires_grad = False
    def forward(
        self,
        prediction: LVSPM,
        batch: BatchedExample,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        loss = self.lpips.forward(
            rearrange(prediction.color, "b v c h w -> (b v) c h w"),
            rearrange(image, "b v c h w -> (b v) c h w"),
            normalize=True,
        )
        return self.cfg.weight * loss.mean()
