import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from model_utils import LectinClassifier
from PIL import Image


class MorphometryAnalyzer:
    def __init__(self):
        # Optical Density conversion vectors for H&E DAB
        # Standard vectors from Ruifrok and Johnston
        # Order: R, G, B
        # Order: R, G, B
        self.He = np.array([0.650, 0.704, 0.286])
        # Customized DAB vector for user sample (more reddish/grayish brown)
        # Original: [0.268, 0.570, 0.776]
        self.Dab = np.array([0.390, 0.530, 0.730])
        self.Res = np.array([0.711, 0.423, 0.561])  # Normalized residual

        # Combined matrix
        self.M = np.array([self.He, self.Dab, self.Res]).T
        try:
            self.Minv = np.linalg.inv(self.M)
        except np.linalg.LinAlgError:
            self.Minv = None
            print("Error: Could not invert stain matrix.")

        # AI Model State
        self.model_ai = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Preprocessing for AI (Must match training)
        self.ai_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_image(self, path):
        """Loads an image from path."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image at {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def segment_tissue(self, image, threshold=180):
        """
        Segments biological tissue from the background.
        Assumes bright/white background.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold: tissue is darker than background
        # Otsu's binarization can be good, or a fixed high threshold for white background
        ret, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def separate_stains(self, image):
        """
        Performs Colour Deconvolution to separate Hematoxylin and DAB.
        Returns H and DAB channels.
        """
        # Convert to float and add epsilon to avoid log(0)
        img_float = image.astype(np.float32) + 1.0

        # Optical Density (OD) conversion: OD = -log10(I / I0)
        # Assuming 8-bit image, I0 = 255
        OD = -np.log10(img_float / 255.0)

        # Reshape to (pixel_count, 3)
        h, w, c = image.shape
        OD_reshaped = OD.reshape((-1, 3))

        # Deconvolution: C = OD * Minv
        # Dimensions: (Pixels, 3) * (3, 3) = (Pixels, 3)
        # Result channels: Columns 0 -> H, 1 -> DAB, 2 -> Residual
        if self.Minv is None:
            return None, None

        C = np.dot(OD_reshaped, self.Minv)

        # Reshape back to image dimensions
        C_reshaped = C.reshape((h, w, 3))

        # Extract channels
        # Swapped channels based on user feedback (Blue being detected as DAB)
        H_channel = C_reshaped[:, :, 1]
        DAB_channel = C_reshaped[:, :, 0]

        # Normalize/Scale back to 0-255 range for visualization/thresholding if needed
        # But usually we work with OD values or normalized versions.
        # For simple thresholding, we can keep OD or normalize.
        # Let's simple clip negative values and process.

        return H_channel, DAB_channel

    def segment_positive_area(self, dab_channel, threshold=0.2):
        """
        Identifies positive (DAB) areas using a threshold on the Deconvolved DAB channel.
        This threshold might need tuning.
        """
        # dab_channel contains OD values. Higher OD = more stain.
        # Create binary mask
        mask = (dab_channel > threshold).astype(np.uint8) * 255

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def calculate_ratio(self, tissue_mask, positive_mask):
        """
        Calculates the percentage of positive area within the tissue.
        """
        # Ensure masks are binary (0 and 255 or 0 and 1)
        # We count non-zero pixels

        # Intersection: Positive pixels that are also inside the tissue
        # (Technically positive mask should already be inside tissue, but good to be safe)
        valid_positive = cv2.bitwise_and(positive_mask, positive_mask, mask=tissue_mask)

        positive_area = cv2.countNonZero(valid_positive)
        total_tissue_area = cv2.countNonZero(tissue_mask)

        if total_tissue_area == 0:
            return 0.0

        ratio = (positive_area / total_tissue_area) * 100.0
        return ratio

    def generate_overlay(self, image, positive_mask, border_mask=None):
        """
        Draws contours of the positive mask on the original image for visualization.
        Optional: Draws border mask contours in Cyan.
        """
        # Find contours
        contours, _ = cv2.findContours(
            positive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw on a copy
        overlay = image.copy()

        # Draw Border if provided (in Blue: R=0, G=0, B=255 for RGB image)
        if border_mask is not None:
            b_contours, _ = cv2.findContours(
                border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, b_contours, -1, (0, 0, 255), 2)

        # Draw filled contours with some transparency or just edges
        # Just edges in Green for high contrast against Brown/Blue
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        return overlay

    def analyze_zonal_intensity(self, tissue_mask, dab_channel, positive_mask):
        """
        Analyzes DAB intensity in Border (Edge) vs Inner regions.
        Consider only pixels that are positive (in positive_mask).
        Returns a dictionary with scores and mean intensities.
        Scale: 0 (Neg), 1 (Weak), 2 (Mod), 3 (Strong)
        """
        # 1. Define Zones
        # Erosion kernel size determines border thickness (~20px for high res)
        kernel_size = 20
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # Border mask = Morphological Gradient (Perimeter)
        border_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_GRADIENT, kernel)

        # Inner mask = Tissue - Border
        inner_mask = cv2.subtract(tissue_mask, border_mask)

        # Debug mask sizes
        print(f"DEBUG: Tissue pixels: {cv2.countNonZero(tissue_mask)}")
        print(f"DEBUG: Inner pixels: {cv2.countNonZero(inner_mask)}")
        print(f"DEBUG: Border pixels: {cv2.countNonZero(border_mask)}")

        # 2. Calculate Intensity Scores
        def get_score(mask, channel, percentile=0.75):
            # Combine zone mask with positive mask
            # ONLY consider pixels that are both in the ZONE and POSITIVE
            combined_mask = cv2.bitwise_and(mask, positive_mask)

            # Mask channel
            masked_channel = cv2.bitwise_and(channel, channel, mask=combined_mask)

            # Extract non-zero pixels (pixels inside combined mask)
            pixels = channel[combined_mask > 0]

            if pixels.size == 0:
                print("DEBUG: No positive pixels in zone mask")
                return 0, 0.0

            valid_pixels = pixels  # Already filtered by positive_mask logic (which implies > threshold)

            if valid_pixels.size == 0:
                mean_intensity = 0.0
            else:
                # Use Top X% of intensities based on zone type
                # Sort pixels ascending
                sorted_pixels = np.sort(valid_pixels)
                # Take the last (1-percentile)%
                # e.g. for Top 10%, percentile should be 0.90
                start_index = int(len(sorted_pixels) * percentile)
                top_pixels = sorted_pixels[start_index:]

                if top_pixels.size == 0:
                    mean_intensity = np.mean(valid_pixels)  # Fallback to all valid
                else:
                    mean_intensity = np.mean(top_pixels)

            # Classification Thresholds (OD) - Calibrated for filtering floor of 0.25
            # Scale shifted to accommodate the fact that we ignore low values
            # 0: < 0.10
            # 1: 0.10 - 0.30 (Expanded to catch the "filtered" low range)
            # 2: 0.30 - 0.42 (Moderate center)
            # 3: > 0.42 (Truly high)
            if mean_intensity < 0.10:
                score = 0
            elif mean_intensity < 0.30:
                score = 1
            elif mean_intensity < 0.42:
                score = 2
            else:
                score = 3

            return score, mean_intensity

        # Calculate for both zones with different strategies
        # Border: Peak detection (Top 15%) for focal intense staining
        border_score, border_mean = get_score(border_mask, dab_channel, percentile=0.85)

        # Inner: Average detection (ALL positive pixels -> Mean) for diffuse staining
        # percentile 0.0 means take top 100% (all sorted pixels)
        inner_score, inner_mean = get_score(inner_mask, dab_channel, percentile=0.0)

        return {
            "border": {"score": border_score, "mean_od": border_mean},
            "inner": {"score": inner_score, "mean_od": inner_mean},
            "masks": {"border": border_mask},  # Return mask for visualization
        }

    def load_ai_model(self, model_path):
        """Loads the trained PyTorch model."""
        if not os.path.exists(model_path):
            return False

        try:
            self.model_ai = LectinClassifier()
            self.model_ai.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model_ai.to(self.device)
            self.model_ai.eval()
            print(f"AI Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading AI model: {e}")
            return False

    def predict_intensity_ai(self, image):
        """Standardizes image and predicts Border/Inner scores using AI."""
        if self.model_ai is None:
            return None

        # Convert numpy (RGB) to PIL for transforms
        pil_img = Image.fromarray(image)
        input_tensor = self.ai_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            b_out, i_out = self.model_ai(input_tensor)
            _, pred_b = torch.max(b_out, 1)
            _, pred_i = torch.max(i_out, 1)

        return {"border_score": int(pred_b.item()), "inner_score": int(pred_i.item())}
