"""
CADabra Mask Utility Nodes
- MaskAnalyzer: Analyze mask statistics (min, max, percentiles, outliers)
- MaskNormalizer: Normalize masks to 0-1 range
- MasksToRGB: Combine 3 masks into RGB image
"""

import torch
import numpy as np


class MaskAnalyzer:
    """
    Analyze mask statistics: min, max, mean, std, percentiles.
    Useful for determining normalization parameters.
    Outputs stats as floats for use in MaskNormalizer manual mode.
    Also outputs a formatted text summary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "percentile_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.5,
                                             "tooltip": "Lower percentile for outlier detection"}),
                "percentile_high": ("FLOAT", {"default": 99.0, "min": 50.0, "max": 100.0, "step": 0.5,
                                              "tooltip": "Upper percentile for outlier detection"}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("stats_text", "min", "max", "mean", "std", "p_low", "p_high", "range", "valid_count")
    OUTPUT_NODE = True
    FUNCTION = "analyze"
    CATEGORY = "CADabra/Utils"

    def analyze(self, mask, percentile_low=1.0, percentile_high=99.0):
        # Convert to numpy for percentile calculation
        data = mask.numpy().flatten()

        # Filter out NaN and invalid values
        valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            # No valid data
            stats_text = "No valid data (all NaN)"
            return {"ui": {"text": [stats_text]}, "result": (stats_text, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)}

        min_val = float(np.min(valid_data))
        max_val = float(np.max(valid_data))
        mean_val = float(np.mean(valid_data))
        std_val = float(np.std(valid_data))
        p_low = float(np.percentile(valid_data, percentile_low))
        p_high = float(np.percentile(valid_data, percentile_high))
        range_val = max_val - min_val
        valid_count = len(valid_data)

        # Format stats text for UI display
        stats_text = (
            f"Min: {min_val:.4f}  Max: {max_val:.4f}\n"
            f"Range: {range_val:.4f}\n"
            f"Mean: {mean_val:.4f}  Std: {std_val:.4f}\n"
            f"P{percentile_low:.0f}: {p_low:.4f}  P{percentile_high:.0f}: {p_high:.4f}\n"
            f"Valid pixels: {valid_count:,}"
        )

        # Print stats to console for visibility
        print(f"[MaskAnalyzer] {stats_text.replace(chr(10), ' | ')}")

        return {"ui": {"text": (stats_text,)}, "result": (stats_text, min_val, max_val, mean_val, std_val, p_low, p_high, range_val, valid_count)}


class MaskNormalizer:
    """
    Normalize a mask to 0-1 range.
    Supports auto-detection of range or manual specification.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "mode": (["auto", "manual"], {"default": "auto",
                                              "tooltip": "auto=detect min/max from data, manual=use specified values"}),
                "min_val": ("FLOAT", {"default": 0.0, "step": 0.01,
                                      "tooltip": "Min value for manual mode"}),
                "max_val": ("FLOAT", {"default": 1.0, "step": 0.01,
                                      "tooltip": "Max value for manual mode"}),
                "clamp": ("BOOLEAN", {"default": True,
                                      "tooltip": "Clamp output to 0-1 range"}),
                "clamp_percentile": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Clamp to percentile range before normalizing. -1=disabled, 0.5=P0.5-P99.5, 1=P1-P99"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "normalize"
    CATEGORY = "CADabra/Utils"

    def normalize(self, mask, mode="auto", min_val=0.0, max_val=1.0, clamp=True, clamp_percentile=-1.0):
        result = mask.clone()

        # Get valid (non-NaN) values
        valid_mask = ~torch.isnan(result)

        # Apply percentile clamping before normalization
        if clamp_percentile >= 0 and valid_mask.any():
            valid_values = result[valid_mask]
            p_low = torch.quantile(valid_values, clamp_percentile / 100.0).item()
            p_high = torch.quantile(valid_values, 1.0 - clamp_percentile / 100.0).item()
            if p_high > p_low:
                result = torch.clamp(result, p_low, p_high)
                print(f"[MaskNormalizer] Clamped to P{clamp_percentile}-P{100-clamp_percentile}: [{p_low:.4f}, {p_high:.4f}]")

        if mode == "auto":
            # Auto-detect range from (potentially clamped) valid values
            valid_mask = ~torch.isnan(result)
            if valid_mask.any():
                valid_values = result[valid_mask]
                min_val = valid_values.min().item()
                max_val = valid_values.max().item()

        # Normalize
        if max_val > min_val:
            result = (result - min_val) / (max_val - min_val)
        else:
            # Constant value, set to 0
            result = torch.zeros_like(result)

        # Handle NaN values (set to 0)
        result = torch.nan_to_num(result, nan=0.0)

        if clamp:
            result = torch.clamp(result, 0.0, 1.0)

        return (result,)


class MasksToRGB:
    """
    Combine 3 masks into an RGB image.
    When auto_normalize is False (default), masks must be pre-normalized to 0-1.
    When auto_normalize is True, each channel is auto-normalized independently.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "red": ("MASK",),
                "green": ("MASK",),
                "blue": ("MASK",),
            },
            "optional": {
                "auto_normalize": ("BOOLEAN", {"default": False,
                                               "tooltip": "Auto-normalize each channel to 0-1 range"}),
                "clamp_percentile": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Clamp each channel to percentile range before normalizing. -1=disabled, 0.5=P0.5-P99.5, 1=P1-P99"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "combine"
    CATEGORY = "CADabra/Utils"

    def _normalize_channel(self, mask, clamp_percentile=-1.0):
        """Normalize a single channel to 0-1 range with optional percentile clamping."""
        result = mask.clone()
        valid_mask = ~torch.isnan(result)
        if valid_mask.any():
            valid_values = result[valid_mask]

            # Apply percentile clamping before normalization
            if clamp_percentile >= 0:
                p_low = torch.quantile(valid_values, clamp_percentile / 100.0).item()
                p_high = torch.quantile(valid_values, 1.0 - clamp_percentile / 100.0).item()
                if p_high > p_low:
                    result = torch.clamp(result, p_low, p_high)
                    valid_values = result[valid_mask]

            min_val = valid_values.min().item()
            max_val = valid_values.max().item()
            if max_val > min_val:
                result = (result - min_val) / (max_val - min_val)
            else:
                result = torch.zeros_like(result)
        result = torch.nan_to_num(result, nan=0.0)
        return torch.clamp(result, 0.0, 1.0)

    def combine(self, red, green, blue, auto_normalize=False, clamp_percentile=-1.0):
        # Handle NaN in all cases
        red = torch.nan_to_num(red, nan=0.0)
        green = torch.nan_to_num(green, nan=0.0)
        blue = torch.nan_to_num(blue, nan=0.0)

        if auto_normalize:
            # Auto-normalize each channel independently
            red = self._normalize_channel(red, clamp_percentile)
            green = self._normalize_channel(green, clamp_percentile)
            blue = self._normalize_channel(blue, clamp_percentile)
        else:
            # Validate all masks are in 0-1 range
            for name, mask in [("red", red), ("green", green), ("blue", blue)]:
                mask_min = mask.min().item()
                mask_max = mask.max().item()
                if mask_min < -0.001 or mask_max > 1.001:  # Small tolerance for float precision
                    raise ValueError(
                        f"{name} mask must be normalized to 0-1 range. "
                        f"Got range [{mask_min:.3f}, {mask_max:.3f}]. "
                        f"Enable auto_normalize or use MaskNormalizer node first."
                    )

        # Ensure all masks have same shape
        if red.shape != green.shape or red.shape != blue.shape:
            raise ValueError(
                f"All masks must have the same shape. "
                f"Got red={red.shape}, green={green.shape}, blue={blue.shape}"
            )

        # Stack into RGB image
        # MASK is (B, H, W), IMAGE is (B, H, W, C)
        image = torch.stack([red, green, blue], dim=-1)

        # Clamp to ensure valid range
        image = torch.clamp(image, 0.0, 1.0)

        return (image,)


class ScrambleFaceIDMask:
    """
    Scramble face ID mask values so adjacent IDs get very different values.
    Uses golden ratio distribution for maximum contrast between sequential face IDs.
    Outputs a MASK with values 0-1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_id_mask": ("MASK",),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31-1,
                    "tooltip": "Random seed for value scrambling"
                }),
                "background_value": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Value for invalid/background pixels (face_id < 0)"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "scramble"
    CATEGORY = "CADabra/Utils"

    def _generate_scrambled_values(self, num_values, seed=42):
        """Generate scrambled values using golden ratio distribution."""
        import random

        rng = random.Random(seed)
        golden_ratio = 0.618033988749895
        value = rng.random()

        values = []
        for i in range(num_values):
            values.append(value)
            value = (value + golden_ratio) % 1.0

        # Shuffle to break any remaining patterns
        rng.shuffle(values)
        return values

    def scramble(self, face_id_mask, seed=42, background_value=0.0):
        import numpy as np

        # Get face IDs as numpy array
        face_ids = face_id_mask.squeeze().cpu().numpy().astype(np.int32)
        H, W = face_ids.shape

        # Find unique face IDs (excluding negative values for invalid/background)
        valid_mask = face_ids >= 0
        unique_ids = np.unique(face_ids[valid_mask])

        if len(unique_ids) == 0:
            # No valid face IDs, return background
            output = np.full((H, W), background_value, dtype=np.float32)
            result = torch.from_numpy(output).unsqueeze(0).float()
            return (result,)

        # Generate enough scrambled values for max face ID
        max_id = int(unique_ids.max())
        scrambled_values = self._generate_scrambled_values(max_id + 1, seed)
        values_array = np.array(scrambled_values, dtype=np.float32)

        # Create output mask with background value
        output = np.full((H, W), background_value, dtype=np.float32)

        # Vectorized mapping: for each unique face ID, assign its scrambled value
        for fid in unique_ids:
            mask = face_ids == fid
            output[mask] = values_array[fid]

        print(f"[ScrambleFaceID] Mapped {len(unique_ids)} unique face IDs to scrambled values")

        # Convert to torch tensor (B, H, W)
        result = torch.from_numpy(output).unsqueeze(0).float()
        return (result,)


NODE_CLASS_MAPPINGS = {
    "MaskAnalyzer": MaskAnalyzer,
    "MaskNormalizer": MaskNormalizer,
    "MasksToRGB": MasksToRGB,
    "ScrambleFaceIDMask": ScrambleFaceIDMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskAnalyzer": "Mask Analyzer",
    "MaskNormalizer": "Mask Normalizer",
    "MasksToRGB": "Masks to RGB",
    "ScrambleFaceIDMask": "Scramble Face ID Mask",
}
