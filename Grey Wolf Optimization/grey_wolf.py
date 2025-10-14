import numpy as np
import cv2
from skimage import exposure
from skimage.filters import sobel
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

class GreyWolfOptimizer:
    """
    Grey Wolf Optimization for Image Enhancement
    Optimizes parameters for image transformation functions
    """
    
    def __init__(self, n_wolves=10, max_iter=50, dim=3):
        """
        Initialize GWO parameters
        
        Args:
            n_wolves: Number of search agents (wolves)
            max_iter: Maximum iterations
            dim: Dimension of search space (number of parameters to optimize)
        """
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = dim
        
        # Initialize positions
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float('-inf')
        
        self.beta_pos = np.zeros(dim)
        self.beta_score = float('-inf')
        
        self.delta_pos = np.zeros(dim)
        self.delta_score = float('-inf')
        
        self.convergence_curve = []
    
    def fitness_function(self, params, image):
        """
        Calculate fitness based on image quality metrics
        
        Args:
            params: [alpha, beta, gamma] for image enhancement
            image: Input grayscale image
            
        Returns:
            fitness_score: Combined fitness value
        """
        alpha, beta, gamma = params
        
        # Apply enhancement transformation
        enhanced = self.enhance_image(image, alpha, beta, gamma)
        
        # Calculate metrics
        entropy = self.calculate_entropy(enhanced)
        edge_content = self.calculate_edge_content(enhanced)
        contrast = self.calculate_contrast(enhanced)
        
        # Combined fitness (weighted sum)
        fitness = 0.4 * entropy + 0.3 * edge_content + 0.3 * contrast
        
        return fitness
    
    def enhance_image(self, image, alpha, beta, gamma):
        """
        Apply enhancement transformation
        
        Args:
            image: Input image
            alpha: Contrast control (0.5 to 3.0)
            beta: Brightness control (-100 to 100)
            gamma: Gamma correction (0.5 to 2.5)
            
        Returns:
            enhanced: Enhanced image
        """
        # Normalize image to [0, 1]
        img_norm = image.astype(np.float32) / 255.0
        
        # Apply contrast and brightness
        adjusted = np.clip(alpha * img_norm + beta / 100.0, 0, 1)
        
        # Apply gamma correction
        enhanced = np.power(adjusted, gamma)
        
        # Convert back to [0, 255]
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def calculate_entropy(self, image):
        """Calculate Shannon entropy"""
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zero entries
        entropy = -np.sum(hist * np.log2(hist))
        return entropy / 8.0  # Normalize to [0, 1]
    
    def calculate_edge_content(self, image):
        """Calculate edge intensity using Sobel operator"""
        img_norm = image.astype(np.float32) / 255.0
        edges = sobel(img_norm)
        edge_content = np.mean(edges)
        return edge_content
    
    def calculate_contrast(self, image):
        """Calculate RMS contrast"""
        mean = np.mean(image)
        std = np.std(image)
        contrast = std / (mean + 1e-10)
        return min(contrast / 2.0, 1.0)  # Normalize
    
    def optimize(self, image):
        """
        Run GWO optimization
        
        Args:
            image: Input grayscale image
            
        Returns:
            best_params: Optimized parameters [alpha, beta, gamma]
        """
        # Define search space bounds
        lb = np.array([0.5, -100, 0.5])  # Lower bounds
        ub = np.array([3.0, 100, 2.5])   # Upper bounds
        
        # Initialize wolf positions randomly
        positions = np.random.uniform(0, 1, (self.n_wolves, self.dim))
        positions = lb + positions * (ub - lb)
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            # Calculate fitness for all wolves
            for i in range(self.n_wolves):
                fitness = self.fitness_function(positions[i], image)
                
                # Update Alpha, Beta, Delta
                if fitness > self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness
                    self.alpha_pos = positions[i].copy()
                elif fitness > self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness
                    self.beta_pos = positions[i].copy()
                elif fitness > self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = positions[i].copy()
            
            # Update a (linearly decreases from 2 to 0)
            a = 2 - iteration * (2.0 / self.max_iter)
            
            # Update positions of all wolves
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    # Update based on Alpha
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # Update based on Beta
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # Update based on Delta
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # Update position
                    positions[i, j] = (X1 + X2 + X3) / 3.0
                
                # Boundary checking
                positions[i] = np.clip(positions[i], lb, ub)
            
            # Store convergence
            self.convergence_curve.append(self.alpha_score)
            
            print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.alpha_score:.4f}")
        
        return self.alpha_pos


def main():
    """Main function to demonstrate GWO for image enhancement"""
    
    # Load image (replace with your image path)
    # For demonstration, create a sample low-contrast image
    print("Creating sample image...")
    sample_image = np.random.randint(80, 150, (256, 256), dtype=np.uint8)
    
    # Or load your own image:
    # image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)
    
    image = sample_image
    
    print(f"Image shape: {image.shape}")
    print(f"Image range: [{image.min()}, {image.max()}]")
    
    # Initialize and run GWO
    print("\nRunning Grey Wolf Optimization...")
    gwo = GreyWolfOptimizer(n_wolves=10, max_iter=30, dim=3)
    best_params = gwo.optimize(image)
    
    print(f"\nOptimization complete!")
    print(f"Best parameters found:")
    print(f"  Alpha (contrast): {best_params[0]:.3f}")
    print(f"  Beta (brightness): {best_params[1]:.3f}")
    print(f"  Gamma (gamma correction): {best_params[2]:.3f}")
    
    # Apply best parameters to enhance image
    enhanced_image = gwo.enhance_image(image, *best_params)
    
    # Calculate improvement metrics
    original_entropy = gwo.calculate_entropy(image)
    enhanced_entropy = gwo.calculate_entropy(enhanced_image)
    
    print(f"\nImage Quality Metrics:")
    print(f"  Original Entropy: {original_entropy:.4f}")
    print(f"  Enhanced Entropy: {enhanced_entropy:.4f}")
    print(f"  Improvement: {((enhanced_entropy - original_entropy) / original_entropy * 100):.2f}%")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Enhanced image
    plt.subplot(2, 3, 2)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title('Enhanced Image (GWO)')
    plt.axis('off')
    
    # Standard histogram equalization for comparison
    hist_eq = cv2.equalizeHist(image)
    plt.subplot(2, 3, 3)
    plt.imshow(hist_eq, cmap='gray')
    plt.title('Histogram Equalization')
    plt.axis('off')
    
    # Original histogram
    plt.subplot(2, 3, 4)
    plt.hist(image.flatten(), bins=256, range=(0, 255), color='blue', alpha=0.7)
    plt.title('Original Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    # Enhanced histogram
    plt.subplot(2, 3, 5)
    plt.hist(enhanced_image.flatten(), bins=256, range=(0, 255), color='green', alpha=0.7)
    plt.title('Enhanced Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    # Convergence curve
    plt.subplot(2, 3, 6)
    plt.plot(gwo.convergence_curve, linewidth=2)
    plt.title('GWO Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gwo_image_enhancement_results.png', dpi=300, bbox_inches='tight')
    print("\nResults saved as 'gwo_image_enhancement_results.png'")
    plt.show()
    
    # Save enhanced image
    cv2.imwrite('enhanced_image.png', enhanced_image)
    print("Enhanced image saved as 'enhanced_image.png'")


if __name__ == "__main__":
    main()