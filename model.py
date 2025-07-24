import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn import functional as F

def correct_transparent_color(solution_rgb, background_rgb, alpha_estimate=0.7):
    """Original correction method using alpha blending."""
    solution_rgb = np.array(solution_rgb, dtype=float)
    background_rgb = np.array(background_rgb, dtype=float)
    white_bg = np.array([255, 255, 255], dtype=float)
    
    corrected_color = (solution_rgb - (1 - alpha_estimate) * background_rgb) / alpha_estimate
    final_color = alpha_estimate * corrected_color + (1 - alpha_estimate) * white_bg
    final_color = np.clip(final_color, 0, 255)
    
    return final_color.astype(int)

def correct_transparent_color_beer_lambert(solution_rgb, background_rgb, path_length=1.0):
    """Beer-Lambert Law correction method."""
    solution_rgb = np.array(solution_rgb, dtype=float) / 255.0
    background_rgb = np.array(background_rgb, dtype=float) / 255.0
    white_bg = np.array([1.0, 1.0, 1.0])
    
    background_rgb = np.maximum(background_rgb, 0.001)
    transmission = solution_rgb / background_rgb
    transmission = np.clip(transmission, 0.001, 1.0)
    
    color_on_white = white_bg * (transmission ** path_length)
    final_color = (color_on_white * 255).astype(int)
    
    return final_color

def extract_enhanced_features(image_path, background_color, center_ratio=0.3):
    """Extract enhanced features including lighting, contrast, and spatial information."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # 1. Original center region average (existing)
    center_h, center_w = int(h * center_ratio), int(w * center_ratio)
    start_h, start_w = (h - center_h) // 2, (w - center_w) // 2
    center_region = img_rgb[start_h:start_h + center_h, start_w:start_w + center_w]
    center_pixels = center_region.reshape(-1, 3).astype(float)
    
    # Remove outliers from center
    mean_color = np.mean(center_pixels, axis=0)
    distances = np.sqrt(np.sum((center_pixels - mean_color) ** 2, axis=1))
    threshold = np.mean(distances) + 2.0 * np.std(distances)
    valid_pixels = center_pixels[distances <= threshold]
    if len(valid_pixels) == 0:
        valid_pixels = center_pixels
    center_avg = np.mean(valid_pixels, axis=0).astype(int)
    
    # 2. Whole image average RGB
    whole_img_avg = np.mean(img_rgb.reshape(-1, 3), axis=0).astype(int)
    
    # 3. Lighting features
    # Convert to LAB color space for better lighting analysis
    img_lab = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    lightness = img_lab[:, :, 0]  # L channel
    
    lighting_features = [
        np.mean(lightness),           # Average lightness
        np.std(lightness),            # Lightness variation
        np.max(lightness),            # Maximum lightness
        np.min(lightness),            # Minimum lightness
        np.percentile(lightness, 75) - np.percentile(lightness, 25)  # Lightness IQR
    ]
    
    # 4. Color contrast features
    # Calculate contrast in each color channel
    contrast_features = []
    for channel in range(3):
        channel_data = img_rgb[:, :, channel]
        contrast = np.std(channel_data)  # Standard deviation as contrast measure
        contrast_features.append(contrast)
    
    # Overall contrast (luminance-based)
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    overall_contrast = np.std(gray)
    contrast_features.append(overall_contrast)
    
    # 5. Spatial gradient features (edge information)
    gradient_features = []
    for channel in range(3):
        channel_data = img_rgb[:, :, channel].astype(np.float32)
        grad_x = cv2.Sobel(channel_data, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel_data, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_features.extend([
            np.mean(gradient_magnitude),  # Average gradient
            np.std(gradient_magnitude)    # Gradient variation
        ])
    
    # 6. Center vs edge comparison
    # Extract edge regions (corners)
    edge_size = min(h, w) // 8
    corners = [
        img_rgb[:edge_size, :edge_size],           # Top-left
        img_rgb[:edge_size, -edge_size:],          # Top-right
        img_rgb[-edge_size:, :edge_size],          # Bottom-left
        img_rgb[-edge_size:, -edge_size:]          # Bottom-right
    ]
    
    edge_avg = np.mean([np.mean(corner.reshape(-1, 3), axis=0) for corner in corners], axis=0)
    center_edge_diff = center_avg - edge_avg  # Difference between center and edges
    
    # 7. Color saturation features
    img_hsv = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = img_hsv[:, :, 2]  # V channel (brightness/saturation)
    saturation_features = [
        np.mean(saturation),
        np.std(saturation),
        np.mean(saturation[start_h:start_h + center_h, start_w:start_w + center_w])  # Center saturation
    ]
    
    # 8. Color uniformity in center region
    center_uniformity = []
    for channel in range(3):
        center_channel = center_region[:, :, channel]
        uniformity = 1.0 / (1.0 + np.std(center_channel))  # Higher value = more uniform
        center_uniformity.append(uniformity)
    
    # Combine all features
    enhanced_features = np.concatenate([
        center_avg,                    # 3 features: center RGB
        whole_img_avg,                 # 3 features: whole image RGB
        lighting_features,             # 5 features: lighting analysis
        contrast_features,             # 4 features: contrast measures
        gradient_features,             # 6 features: spatial gradients
        center_edge_diff,              # 3 features: center vs edge
        saturation_features,           # 3 features: saturation
        center_uniformity              # 3 features: center uniformity
    ])
    
    return center_avg.astype(int), enhanced_features.astype(np.float32)

class ConcentrationDataset(Dataset):
    """Dataset class for loading images and concentration labels with dual-region processing."""
    
    def __init__(self, image_paths, concentrations, color_features, transform=None, image_size=224):
        self.image_paths = image_paths
        self.concentrations = concentrations
        self.color_features = color_features
        self.image_size = image_size
        
        # Standard transform for full image (to see both center and corners)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load full image (contains both solution center and background corners)
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms to full image
        image_tensor = self.transform(image)
        
        # Get concentration and color features
        concentration = torch.tensor(self.concentrations[idx], dtype=torch.float32)
        color_feat = torch.tensor(self.color_features[idx], dtype=torch.float32)
        
        return image_tensor, color_feat, concentration

class CNNFeatureExtractor(nn.Module):
    """CNN with dual-region attention for solution center + background corners."""
    
    def __init__(self, feature_dim=64, use_pretrained=False):
        super(CNNFeatureExtractor, self).__init__()
        
        if use_pretrained:
            self.backbone = models.mobilenet_v2(pretrained=True)
            self.backbone.classifier = nn.Identity()
            backbone_out = 1280
        else:
            # Custom CNN that processes full image
            self.backbone = nn.Sequential(
                # First conv block - learn basic features
                nn.Conv2d(3, 32, 7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Second conv block - spatial relationships
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Third conv block - high level features
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
            backbone_out = 128
        
        # Dual-region attention: emphasize center but also consider corners
        self.center_attention = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.background_attention = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Separate processing for center and background features
        self.center_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.corner_pool = nn.AdaptiveMaxPool2d((2, 2))  # 2x2 to capture corners
        
        # Feature fusion
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out + backbone_out * 4, feature_dim),  # Center + 4 corners
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # Extract features from backbone
        backbone_features = self.backbone(x)  # Shape: [batch, 128, H, W]
        
        # Generate attention maps
        center_attn = self.center_attention(backbone_features)  # Focus on center
        bg_attn = self.background_attention(backbone_features)   # Focus on background
        
        # Apply attention to features
        center_features = backbone_features * center_attn
        bg_features = backbone_features * bg_attn
        
        # Pool features differently for center vs background
        center_pooled = self.center_pool(center_features).flatten(1)  # Global average for center
        bg_pooled = self.corner_pool(bg_features).flatten(1)          # 2x2 max pool for corners
        
        # Combine center and background features
        combined_features = torch.cat([center_pooled, bg_pooled], dim=1)
        
        # Apply feature extraction
        features = self.feature_extractor(combined_features)
        
        return features

class EnsembleConcentrationModel(nn.Module):
    """Lightweight ensemble model combining CNN features with color correction features."""
    
    def __init__(self, color_feature_dim, cnn_feature_dim=64, fusion_dim=32, use_pretrained=False):
        super(EnsembleConcentrationModel, self).__init__()
        
        # CNN feature extractor (much smaller)
        self.cnn_extractor = CNNFeatureExtractor(feature_dim=cnn_feature_dim, use_pretrained=use_pretrained)
        
        # Color feature processor (smaller)
        self.color_processor = nn.Sequential(
            nn.Linear(color_feature_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion network (smaller)
        self.fusion = nn.Sequential(
            nn.Linear(cnn_feature_dim + 16, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 1)
        )
    
    def forward(self, image, color_features):
        # Extract CNN features
        cnn_features = self.cnn_extractor(image)
        
        # Process color features
        color_features = self.color_processor(color_features)
        
        # Fuse features
        combined_features = torch.cat([cnn_features, color_features], dim=1)
        
        # Predict concentration
        concentration = self.fusion(combined_features)
        
        return concentration.squeeze()

class ColorConcentrationEnsemble:
    def __init__(self, data_dir="./Data", background_filename="background.png", device=None):
        self.data_dir = Path(data_dir)
        self.background_path = self.data_dir / background_filename
        
        # Concentration mapping
        self.concentration_map = {
            '1': 0.39, '2': 0.78, '3': 1.56, '4': 3.13,
            '5': 6.25, '6': 12.6, '7': 25.0, '8': 50.0
        }
        
        self.background_color = None
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Models
        self.ensemble_model = None
        self.linear_models = {}
        self.scalers = {}
        self.training_data = None
        
    def load_background(self):
        """Load and process background image."""
        if not self.background_path.exists():
            raise FileNotFoundError(f"Background image not found: {self.background_path}")
        
        self.background_color, self.background_enhanced = extract_enhanced_features(self.background_path, None)
        print(f"Background color (RGB): {self.background_color}")
        print(f"Background enhanced features: {len(self.background_enhanced)} dimensions")
        
    def collect_training_data(self, correction_method='all_combined', alpha_estimate=0.5, path_length=1.0):
        """Collect training data from all folders."""
        if self.background_color is None:
            self.load_background()
        
        training_data = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        print(f"Collecting training data (correction method: {correction_method})...")
        
        for folder_name, concentration in self.concentration_map.items():
            folder_path = self.data_dir / folder_name
            
            if not folder_path.exists():
                print(f"Warning: Folder {folder_path} not found, skipping...")
                continue
            
            print(f"Processing folder {folder_name} (concentration: {concentration})...")
            
            # Get all image files in the folder
            image_files = [f for f in folder_path.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            if not image_files:
                print(f"No images found in folder {folder_name}")
                continue
            
            for img_path in image_files:
                try:
                    # Extract enhanced features including lighting and spatial info
                    original_color, enhanced_features = extract_enhanced_features(img_path, self.background_color)
                    
                    # Apply traditional corrections to center color
                    corrected_alpha = correct_transparent_color(
                        original_color, self.background_color, alpha_estimate
                    )
                    corrected_beer_lambert = correct_transparent_color_beer_lambert(
                        original_color, self.background_color, path_length
                    )
                    
                    # Create comprehensive feature vector
                    if correction_method == 'none':
                        color_features = np.concatenate([
                            original_color, 
                            self.background_color,
                            enhanced_features  # Add enhanced features
                        ])
                    elif correction_method == 'alpha':
                        color_features = np.concatenate([
                            corrected_alpha, 
                            self.background_color,
                            enhanced_features
                        ])
                    elif correction_method == 'beer_lambert':
                        color_features = np.concatenate([
                            corrected_beer_lambert, 
                            self.background_color,
                            enhanced_features
                        ])
                    elif correction_method == 'all_combined':
                        color_features = np.concatenate([
                            original_color,
                            corrected_alpha,
                            corrected_beer_lambert,
                            self.background_color,
                            enhanced_features  # Add all enhanced features
                        ])
                    else:
                        raise ValueError(f"Unknown correction method: {correction_method}")
                    
                    training_data.append({
                        'image_path': str(img_path),
                        'color_features': color_features,
                        'concentration': concentration,
                        'folder': folder_name,
                        'filename': img_path.name,
                        'original_color': original_color,
                        'corrected_alpha': corrected_alpha,
                        'corrected_beer_lambert': corrected_beer_lambert,
                        'enhanced_features': enhanced_features,
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        if not training_data:
            raise ValueError("No training data collected!")
        
        self.training_data = training_data
        print(f"Collected {len(training_data)} training samples")
        print(f"Color feature dimensions: {training_data[0]['color_features'].shape[0]}")
        
        return training_data
    
    def detailed_analysis(self):
        """Perform detailed analysis to understand R² performance."""
        if self.training_data is None:
            print("No training data available.")
            return
        
        # Analyze data quality
        concentrations = [sample['concentration'] for sample in self.training_data]
        original_colors = np.array([sample['original_color'] for sample in self.training_data])
        
        print("🔍 DETAILED ANALYSIS - Why is R² = 0.73?")
        print("="*60)
        
        # 1. Data distribution analysis
        unique_conc, counts = np.unique(concentrations, return_counts=True)
        print(f"📊 Data Distribution:")
        total_var = 0
        for conc, count in zip(unique_conc, counts):
            subset_colors = original_colors[np.array(concentrations) == conc]
            color_std = np.std(subset_colors, axis=0)
            avg_std = np.mean(color_std)
            total_var += avg_std
            print(f"   Conc {conc:5.2f}: {count:3d} samples, Avg color std: {avg_std:.1f}")
        
        avg_variability = total_var / len(unique_conc)
        print(f"   Average color variability within groups: {avg_variability:.1f}")
        
        if avg_variability > 20:
            print("   ⚠️  HIGH variability within concentration groups!")
            print("      → Same concentration = very different colors")
            print("      → This limits max possible R²")
        
        # 2. Concentration range analysis
        conc_range = max(concentrations) - min(concentrations)
        conc_log_range = np.log10(max(concentrations)) - np.log10(min(concentrations))
        print(f"\n📏 Concentration Range:")
        print(f"   Linear range: {min(concentrations):.2f} - {max(concentrations):.2f} ({conc_range:.1f}x)")
        print(f"   Log range: {conc_log_range:.1f} orders of magnitude")
        
        if conc_log_range > 2:
            print("   ⚠️  VERY WIDE range (>100x)!")
            print("      → Consider log-scale modeling")
        
        # 3. Color-concentration correlation
        print(f"\n🎨 Color-Concentration Correlation:")
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            corr = np.corrcoef(concentrations, original_colors[:, i])[0, 1]
            print(f"   {channel:5s} channel: r = {corr:.3f}")
            if abs(corr) < 0.5:
                print(f"      ⚠️  Weak correlation for {channel} channel")
        
        # 4. Outlier detection
        print(f"\n🔍 Outlier Analysis:")
        outlier_count = 0
        for conc in unique_conc:
            subset_colors = original_colors[np.array(concentrations) == conc]
            if len(subset_colors) > 1:
                mean_color = np.mean(subset_colors, axis=0)
                distances = np.sqrt(np.sum((subset_colors - mean_color) ** 2, axis=1))
                outliers = distances > (np.mean(distances) + 2 * np.std(distances))
                outlier_count += np.sum(outliers)
        
        outlier_percentage = (outlier_count / len(self.training_data)) * 100
        print(f"   Potential outliers: {outlier_count}/{len(self.training_data)} ({outlier_percentage:.1f}%)")
        
        if outlier_percentage > 15:
            print("   ⚠️  HIGH outlier rate!")
            print("      → Check image quality, lighting conditions")
        
        # 5. Recommendations
        print(f"\n💡 RECOMMENDATIONS TO IMPROVE R²:")
        print("="*60)
        
        if avg_variability > 20:
            print("1. 🎯 Improve data quality:")
            print("   - Standardize lighting conditions")
            print("   - Use consistent camera settings")
            print("   - Check for mislabeled samples")
        
        if conc_log_range > 2:
            print("2. 📊 Try log-scale modeling:")
            print("   - Use log(concentration) as target")
            print("   - May improve linearity")
        
        if outlier_percentage > 10:
            print("3. 🧹 Clean outliers:")
            print("   - Remove images with poor quality")
            print("   - Check for labeling errors")
        
        print("4. 🤖 Model improvements:")
        print("   - Try different CNN architectures")
        print("   - Add data augmentation")
        print("   - Ensemble multiple models")
        
        print("5. 📸 Collect more data:")
        print("   - More samples per concentration")
        print("   - Better controlled conditions")
        
        return {
            'avg_variability': avg_variability,
            'outlier_percentage': outlier_percentage,
            'conc_log_range': conc_log_range
        }
    
    def train_ensemble_model(self, test_size=0.2, batch_size=8, epochs=60, lr=0.001, use_pretrained=False, 
                           enable_continuous_output=True):
        """Train the ensemble CNN + color correction model."""
        if self.training_data is None:
            raise ValueError("No training data available. Call collect_training_data first.")
        
        # Prepare data
        image_paths = [sample['image_path'] for sample in self.training_data]
        concentrations = [sample['concentration'] for sample in self.training_data]
        color_features = np.array([sample['color_features'] for sample in self.training_data])
        
        # Make training more continuous by adding noise and interpolation
        if enable_continuous_output:
            print("🔄 Enabling continuous output training...")
            
            # Add label smoothing / noise to concentrations to encourage continuous output
            concentrations_noisy = []
            for conc in concentrations:
                # Add smaller random noise (±1% of concentration value)
                noise = np.random.normal(0, 0.01 * conc)  # Reduced from 5% to 1%
                concentrations_noisy.append(conc + noise)
            
            # Create interpolated samples between concentration levels
            original_size = len(image_paths)
            interpolated_paths = []
            interpolated_concentrations = []
            interpolated_features = []
            
            for i in range(len(image_paths) // 4):  # Add 25% more samples
                # Pick two random samples
                idx1, idx2 = np.random.choice(len(image_paths), 2, replace=False)
                
                # Interpolate concentration (between the two samples)
                alpha = np.random.uniform(0.3, 0.7)  # Interpolation weight
                interp_conc = alpha * concentrations[idx1] + (1 - alpha) * concentrations[idx2]
                
                # Use one of the images (can't interpolate images easily)
                interp_path = image_paths[idx1]
                
                # Interpolate color features
                interp_features = alpha * color_features[idx1] + (1 - alpha) * color_features[idx2]
                
                interpolated_paths.append(interp_path)
                interpolated_concentrations.append(interp_conc)
                interpolated_features.append(interp_features)
            
            # Combine original and interpolated data
            image_paths.extend(interpolated_paths)
            concentrations = concentrations_noisy + interpolated_concentrations
            color_features = np.vstack([color_features, np.array(interpolated_features)])
            
            print(f"   Added {len(interpolated_paths)} interpolated samples")
            print(f"   Total samples: {len(image_paths)} (was {original_size})")
        
        print(f"Dataset size: {len(image_paths)} samples")
        print(f"Using {'pretrained' if use_pretrained else 'custom'} CNN backbone")
        
        # Split data
        indices = list(range(len(image_paths)))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # Scale color features
        scaler = StandardScaler()
        color_features_scaled = scaler.fit_transform(color_features)
        self.scalers['color_scaler'] = scaler
        
        # Create datasets with smaller image size
        train_dataset = ConcentrationDataset(
            [image_paths[i] for i in train_idx],
            [concentrations[i] for i in train_idx],
            [color_features_scaled[i] for i in train_idx],
            image_size=128  # Smaller images for faster training
        )
        
        test_dataset = ConcentrationDataset(
            [image_paths[i] for i in test_idx],
            [concentrations[i] for i in test_idx],
            [color_features_scaled[i] for i in test_idx],
            image_size=128
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Initialize model
        color_feature_dim = color_features.shape[1]
        self.ensemble_model = EnsembleConcentrationModel(
            color_feature_dim=color_feature_dim,
            cnn_feature_dim=64,  # Smaller
            fusion_dim=32,       # Smaller
            use_pretrained=use_pretrained
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.ensemble_model.parameters())
        trainable_params = sum(p.numel() for p in self.ensemble_model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Loss function and optimizer WITH COSINE ANNEALING
        if enable_continuous_output:
            criterion = nn.SmoothL1Loss()
            print("Using SmoothL1Loss for continuous training")
        else:
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(self.ensemble_model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Cosine annealing scheduler - smooth LR decay from lr to lr*0.01
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,           # Full cosine cycle over all epochs
            eta_min=lr * 0.01       # Minimum LR = 1% of initial LR
        )
        print(f"Using CosineAnnealingLR: {lr:.6f} → {lr*0.01:.6f} over {epochs} epochs")
        
        # Training loop with progress tracking
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        early_stopping_patience = 20  # Fixed: Define early stopping patience
        
        print(f"Training ensemble model for up to {epochs} epochs...")
        print("Progress: ", end="", flush=True)
        
        for epoch in range(epochs):
            # Training
            self.ensemble_model.train()
            train_loss = 0.0
            batch_count = 0
            
            for images, color_feats, targets in train_loader:
                images = images.to(self.device)
                color_feats = color_feats.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.ensemble_model(images, color_feats)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
            
            # Validation
            self.ensemble_model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for images, color_feats, targets in test_loader:
                    images = images.to(self.device)
                    color_feats = color_feats.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.ensemble_model(images, color_feats)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    val_predictions.extend(outputs.cpu().numpy() if outputs.dim() > 0 else [outputs.cpu().item()])
                    val_targets.extend(targets.cpu().numpy() if targets.dim() > 0 else [targets.cpu().item()])
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update learning rate with scheduler
            scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.ensemble_model.state_dict(), 'best_ensemble_model.pth')
            else:
                patience_counter += 1
            
            # Progress indicator
            if epoch % max(1, epochs // 50) == 0:
                print("█", end="", flush=True)
            
            # Detailed progress every 15 epochs
            if epoch % 15 == 0:
                val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
                val_r2 = r2_score(val_targets, val_predictions)
                current_lr = optimizer.param_groups[0]['lr']  # Get actual current LR
                print(f"\nEpoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}, LR: {current_lr:.6f}")
                print("Progress: ", end="", flush=True)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} (patience exceeded)")
                break
        
        print(" Done!" if patience_counter < early_stopping_patience else "")
        
        # Load best model (this ensures we use the best model, not the last one)
        if os.path.exists('best_ensemble_model.pth'):
            self.ensemble_model.load_state_dict(torch.load('best_ensemble_model.pth'))
            print(f"✅ Loaded best model from epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
        else:
            print("⚠️  No best model file found, using current model state")
        
        # Final evaluation on ALL DATA (not just test set for full picture)
        self.ensemble_model.eval()
        final_predictions = []
        final_targets = []
        train_predictions_final = []
        train_targets_final = []
        
        print("🔍 Evaluating on test set...")
        with torch.no_grad():
            for images, color_feats, targets in test_loader:
                images = images.to(self.device)
                color_feats = color_feats.to(self.device)
                
                outputs = self.ensemble_model(images, color_feats)
                final_predictions.extend(outputs.cpu().numpy() if outputs.dim() > 0 else [outputs.cpu().item()])
                final_targets.extend(targets.cpu().numpy() if targets.dim() > 0 else [targets.cpu().item()])
        
        print("🔍 Evaluating on training set for full visualization...")
        with torch.no_grad():
            for images, color_feats, targets in train_loader:
                images = images.to(self.device)
                color_feats = color_feats.to(self.device)
                
                outputs = self.ensemble_model(images, color_feats)
                train_predictions_final.extend(outputs.cpu().numpy() if outputs.dim() > 0 else [outputs.cpu().item()])
                train_targets_final.extend(targets.cpu().numpy() if targets.dim() > 0 else [targets.cpu().item()])
        
        final_rmse = np.sqrt(mean_squared_error(final_targets, final_predictions))
        final_r2 = r2_score(final_targets, final_predictions)
        final_mae = mean_absolute_error(final_targets, final_predictions)
        
        print(f"\n🎯 Final Ensemble Model Results:")
        print(f"   Test RMSE: {final_rmse:.4f}")
        print(f"   Test MAE: {final_mae:.4f}")
        print(f"   Test R²: {final_r2:.4f}")
        print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        
        # Plot both train and test with ALL points
        plt.scatter(train_targets_final, train_predictions_final, alpha=0.4, 
                   label=f'Train (n={len(train_targets_final)})', s=30, color='lightblue')
        plt.scatter(final_targets, final_predictions, alpha=0.8, 
                   label=f'Test (n={len(final_targets)})', s=40, color='orange')
        
        # Perfect prediction line
        all_targets = train_targets_final + final_targets
        all_predictions = train_predictions_final + final_predictions
        min_val = min(min(all_targets), min(all_predictions))
        max_val = max(max(all_targets), max(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
        
        plt.xlabel('Actual Concentration')
        plt.ylabel('Predicted Concentration')
        plt.title(f'Enhanced Ensemble Model\nTest R² = {final_r2:.4f} | Total Points = {len(all_targets)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add some statistics text
        plt.text(0.05, 0.95, f'Train R² = {r2_score(train_targets_final, train_predictions_final):.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        plt.text(0.05, 0.85, f'Test R² = {final_r2:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        return {
            'final_rmse': final_rmse,
            'final_r2': final_r2,
            'final_mae': final_mae,
            'predictions': final_predictions,
            'targets': final_targets,
            'train_predictions': train_predictions_final,
            'train_targets': train_targets_final,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch,
            'early_stopped': patience_counter >= early_stopping_patience
        }
    
    def train_linear_models(self, test_size=0.2):
        """Train traditional linear models for comparison."""
        if self.training_data is None:
            raise ValueError("No training data available.")
        
        color_features = np.array([sample['color_features'] for sample in self.training_data])
        concentrations = [sample['concentration'] for sample in self.training_data]
        
        X_train, X_test, y_train, y_test = train_test_split(
            color_features, concentrations, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            if name == 'linear':
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'predictions': predictions,
                'targets': y_test
            }
            
            print(f"{name.upper()} Model:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
        
        self.linear_models = models
        self.scalers['linear_scaler'] = scaler
        
        return results
    
    def predict_concentration(self, image_path, method='ensemble'):
        """Predict concentration for a single image."""
        if method == 'ensemble':
            if self.ensemble_model is None:
                raise ValueError("Ensemble model not trained. Call train_ensemble_model first.")
            
            # Prepare image
            transform = transforms.Compose([
                transforms.Resize((128, 128)),  # Match training size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(self.device)
            
            # Extract enhanced color features  
            original_color, enhanced_features = extract_enhanced_features(image_path, self.background_color)
            corrected_alpha = correct_transparent_color(original_color, self.background_color)
            corrected_beer_lambert = correct_transparent_color_beer_lambert(original_color, self.background_color)
            
            color_features = np.concatenate([
                original_color, corrected_alpha, corrected_beer_lambert, 
                self.background_color, enhanced_features
            ])
            
            color_features = self.scalers['color_scaler'].transform(color_features.reshape(1, -1))
            color_features = torch.tensor(color_features, dtype=torch.float32).to(self.device)
            
            # Predict
            self.ensemble_model.eval()
            with torch.no_grad():
                prediction = self.ensemble_model(image, color_features)
            
            return prediction.cpu().item(), original_color
        
        else:
            # Use linear models
            if method not in self.linear_models:
                raise ValueError(f"Linear model '{method}' not found.")
            
            original_color, enhanced_features = extract_enhanced_features(image_path, self.background_color)
            corrected_alpha = correct_transparent_color(original_color, self.background_color)
            corrected_beer_lambert = correct_transparent_color_beer_lambert(original_color, self.background_color)
            
            color_features = np.concatenate([
                original_color, corrected_alpha, corrected_beer_lambert, 
                self.background_color, enhanced_features
            ])
            
            if method == 'linear':
                color_features = self.scalers['linear_scaler'].transform(color_features.reshape(1, -1))
            
            prediction = self.linear_models[method].predict(color_features.reshape(1, -1))[0]
            
            return prediction, original_color

# Example usage
def main():
    # Initialize ensemble trainer
    trainer = ColorConcentrationEnsemble(data_dir="./Data")
    
    # Collect training data
    trainer.collect_training_data(correction_method='all_combined')
    
    # FIRST: Analyze why R² might be low
    print("🔍 DIAGNOSING R² = 0.73...")
    analysis = trainer.detailed_analysis()
    
    print("\nTraining traditional linear models...")
    linear_results = trainer.train_linear_models()
    
    print("\nTraining LIGHTWEIGHT ensemble CNN + color correction model...")
    ensemble_results = trainer.train_ensemble_model(
        epochs=90,
        batch_size=8,
        use_pretrained=False,
        enable_continuous_output=True  # 🔧 Enable continuous output
    )
    
    # Compare results
    print("\n" + "="*60)
    print("🏆 MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Status':<15}")
    print("-" * 60)
    
    for name, result in linear_results.items():
        status = "Good" if result['r2'] > 0.8 else "Needs work"
        print(f"{name:<20} {result['rmse']:<8.3f} {result['mae']:<8.3f} {result['r2']:<8.3f} {status:<15}")
    
    ensemble_status = "Good" if ensemble_results['final_r2'] > 0.8 else "Needs work"
    print(f"{'Ensemble CNN':<20} {ensemble_results['final_rmse']:<8.3f} "
          f"{ensemble_results['final_mae']:<8.3f} {ensemble_results['final_r2']:<8.3f} {ensemble_status:<15}")
    
    # Interpretation
    print(f"\n📊 INTERPRETATION:")
    if ensemble_results['final_r2'] > 0.8:
        print("✅ R² > 0.8: Excellent performance!")
    elif ensemble_results['final_r2'] > 0.7:
        print("⚠️  R² = 0.7-0.8: Decent, but room for improvement")
        print("   → Check the detailed analysis above for specific issues")
    else:
        print("❌ R² < 0.7: Significant issues - data quality or model problems")
    
    return trainer, analysis

def quick_test():
    """Quick test with even smaller model for debugging."""
    print("🚀 QUICK TEST - Ultra-lightweight model")
    trainer = ColorConcentrationEnsemble(data_dir="./Data")
    trainer.collect_training_data(correction_method='none')  # Simplest features
    
    # Train for just 10 epochs to see if it moves
    results = trainer.train_ensemble_model(
        epochs=10,
        batch_size=4,
        use_pretrained=False
    )
    
    print("Quick test completed! If you see this, the model is working.")
    return results

if __name__ == "__main__":
    main()