import numpy as np
import matplotlib.pyplot as plt
from pymanopt.manifolds.hyperbolic import PoincareBall


# ============================================================================
# PROBLEM DEFINITION: Total Variation Denoising on Hyperbolic Space
# ============================================================================

class TVDenoisingProblem(object):
    """
    Total Variation Denoising Problem on (H^2)^n
    
    Minimizes: f_q(p) = (1/n)[g(p,q) + α·TV(p)]
    where:
        - g(p,q) = (1/2)Σ dist²(p[i], q[i])  [data fidelity]
        - TV(p) = Σ dist(p[i], p[i+1])       [total variation]
    """
    
    def __init__(self, q_clean, q_noisy, alpha=0.5):
        """
        Parameters:
        - q_clean: clean signal, shape (n, 3)
        - q_noisy: noisy signal, shape (n, 3)
        - alpha: TV regularization parameter
        """
        self.q_clean = q_clean
        self.q_noisy = q_noisy
        self.alpha = alpha
        self.n = len(q_noisy)
        
        # Create manifolds
        self.manifold_single = PoincareBall(2)  # For single point operations (2D Poincare ball)
        self.manifold_product = PoincareBall(2)  # For full signal
        
        # Compute diameter
        self.diameter = self._compute_diameter()
        
        # Curvature bounds for H^2
        self.omega = -1  # Lower curvature bound
        self.Omega = -1  # Upper curvature bound
    
    @classmethod
    def from_square_wave(cls, T=3, a=-6, b=6, N=496, alpha=0.5, 
                         noise_std=0.1, seed=42):
        """
        Create a TV denoising problem from a square wave signal
        
        Parameters:
        - T: period of square wave
        - a, b: interval bounds
        - N: target number of discretization points
        - alpha: TV regularization parameter
        - noise_std: standard deviation of noise
        - seed: random seed
        
        Returns:
        - TVDenoisingProblem instance
        """
        np.random.seed(seed)
        
        # Create single point manifold for construction
        manifold_single = PoincareBall(2)
        
        # Step 1: Create square wave in R^2
        print(f"Creating square wave with period T={T}, interval [{a}, {b}]...")
        t, p = cls._create_square_wave(T, a, b, N)
        
        # Step 2: Extract jump points
        S_indices = cls._extract_jump_points(t, p, T)
        M = len(S_indices)
        print(f"Jump points: {M}")
        
        # Step 3: Map to hyperbolic space
        s = np.array([cls._phi_map(p[i]) for i in S_indices])
        
        # Step 4: Sample geodesics
        u = int(np.floor(N * T / (2 * (b - a))))
        print(f"Samples per geodesic: {u}")
        
        q_clean = []
        for i in range(M // 2):
            p_start = s[2*i]
            p_end = s[2*i + 1]
            geodesic_points = cls._sample_geodesic(manifold_single, p_start, p_end, u)
            q_clean.extend(geodesic_points)
        
        q_clean = np.array(q_clean)
        n = len(q_clean)
        print(f"Signal length: {n}")
        
        # Step 5: Add Gaussian noise
        q_noisy = np.zeros_like(q_clean)
        for i in range(n):
            noise = np.random.randn(2) * noise_std
            noise = manifold_single.projection(q_clean[i], noise)
            q_noisy[i] = manifold_single.exp(q_clean[i], noise)
        
        # Create and return problem instance
        return cls(q_clean, q_noisy, alpha)
    
    @staticmethod
    def _phi_map(p):
        """Map from R^2 to Poincare ball"""
        p1, p2 = p[0], p[1]
        # Scale down and map to ensure we stay in the ball
        # Simple scaling to keep points well within unit ball
        scale = 0.8  # Keep points away from boundary
        norm = np.sqrt(p1**2 + p2**2)
        if norm > 0:
            factor = scale / (1 + norm)
            return np.array([p1 * factor, p2 * factor])
        else:
            return np.array([0.0, 0.0])
    
    @staticmethod
    def _create_square_wave(T, a, b, N):
        """Create square wave signal in R^2"""
        t = np.linspace(a, b, N)
        p = np.zeros((N, 2))
        p[:, 0] = t
        p[:, 1] = np.sign(np.sin(2 * np.pi * t / T))
        return t, p
    
    @staticmethod
    def _extract_jump_points(t, p, T):
        """Extract points at discontinuities of square wave"""
        p2 = p[:, 1]
        jumps = np.where(np.diff(p2) != 0)[0]
        
        S_indices = [0]
        for jump_idx in jumps:
            S_indices.append(jump_idx)
            S_indices.append(jump_idx + 1)
        S_indices.append(len(t) - 1)
        
        return sorted(list(set(S_indices)))
    
    @staticmethod
    def _sample_geodesic(manifold, p_start, p_end, num_points):
        """Sample points along geodesic from p_start to p_end"""
        points = []

        # Check if points are valid
        if np.linalg.norm(p_start) >= 1.0 or np.linalg.norm(p_end) >= 1.0:
            print(f"Warning: Points outside ball - start: {np.linalg.norm(p_start)}, end: {np.linalg.norm(p_end)}")
            # Project points into ball
            if np.linalg.norm(p_start) >= 1.0:
                p_start = p_start / np.linalg.norm(p_start) * 0.95
            if np.linalg.norm(p_end) >= 1.0:
                p_end = p_end / np.linalg.norm(p_end) * 0.95

        try:
            v = manifold.log(p_start, p_end)
            for j in range(num_points):
                t_param = j / (num_points - 1) if num_points > 1 else 0
                point = manifold.exp(p_start, t_param * v)
                points.append(point)
        except Exception as e:
            print(f"Error in geodesic sampling: {e}")
            # Fallback: linear interpolation in ambient space, then project
            for j in range(num_points):
                t_param = j / (num_points - 1) if num_points > 1 else 0
                point = (1 - t_param) * p_start + t_param * p_end
                # Project to ball if needed
                if np.linalg.norm(point) >= 1.0:
                    point = point / np.linalg.norm(point) * 0.95
                points.append(point)

        return np.array(points)
    
    def _compute_diameter(self):
        """Compute diameter: 3 × max pairwise distance in noisy signal"""
        max_dist = 0
        sample_size = min(self.n, 100)
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                dist = self.manifold_single.dist(self.q_noisy[i], self.q_noisy[j])
                max_dist = max(max_dist, dist)
        return 3 * max_dist
    
    def objective(self, p):
        """
        Compute objective function value
        f_q(p) = (1/n)[g(p,q) + α·TV(p)]
        """
        # Data fidelity term: (1/2)Σ dist²(p[i], q[i])
        data_fidelity = 0.5 * sum(
            self.manifold_single.dist(p[i], self.q_noisy[i])**2 
            for i in range(self.n)
        )
        
        # Total variation term: Σ dist(p[i], p[i+1])
        tv = sum(
            self.manifold_single.dist(p[i], p[i+1]) 
            for i in range(self.n - 1)
        )
        
        return (data_fidelity + self.alpha * tv) / self.n
    
    def subdifferential(self, p):
        """
        Compute a subgradient of the objective function
        ∂f_q(p) = (1/n)[-log_p(q̄) + α·∂TV(p)]

        Returns: array of subgradients, shape (n, 3)
        """
        subgrad = np.zeros_like(p)

        for i in range(self.n):
            # Ensure points are valid (inside the ball)
            p_i = p[i].copy()
            q_i = self.q_noisy[i].copy()

            # Project points to ensure they're strictly inside the ball
            if np.linalg.norm(p_i) >= 0.99:
                p_i = p_i / np.linalg.norm(p_i) * 0.95
            if np.linalg.norm(q_i) >= 0.99:
                q_i = q_i / np.linalg.norm(q_i) * 0.95

            try:
                # Data fidelity gradient: -log_{p[i]}(q_noisy[i])
                data_grad = -self.manifold_single.log(p_i, q_i)

                # Check for NaN/inf in data gradient
                if not np.isfinite(data_grad).all():
                    data_grad = np.zeros(2)
            except:
                data_grad = np.zeros(2)

            # TV subdifferential contribution
            tv_subgrad = np.zeros(2)

            # Contribution from dist(p[i-1], p[i]) if i > 0
            if i > 0:
                p_prev = p[i-1].copy()
                if np.linalg.norm(p_prev) >= 0.99:
                    p_prev = p_prev / np.linalg.norm(p_prev) * 0.95

                try:
                    dist_prev = self.manifold_single.dist(p_prev, p_i)
                    if dist_prev > 1e-8:
                        log_from_prev = self.manifold_single.log(p_i, p_prev)
                        if np.isfinite(log_from_prev).all():
                            tv_subgrad += -log_from_prev / dist_prev
                except:
                    pass

            # Contribution from dist(p[i], p[i+1]) if i < n-1
            if i < self.n - 1:
                p_next = p[i+1].copy()
                if np.linalg.norm(p_next) >= 0.99:
                    p_next = p_next / np.linalg.norm(p_next) * 0.95

                try:
                    dist_next = self.manifold_single.dist(p_i, p_next)
                    if dist_next > 1e-8:
                        log_to_next = self.manifold_single.log(p_i, p_next)
                        if np.isfinite(log_to_next).all():
                            tv_subgrad += -log_to_next / dist_next
                except:
                    pass

            # Combine
            subgrad[i] = (data_grad + self.alpha * tv_subgrad) / self.n

        return subgrad
    
    def initial_point(self):
        """Return initial point for optimization (noisy signal)"""
        return self.q_noisy.copy()
    
    def compute_error(self, p):
        """
        Compute mean error w.r.t. clean signal
        E(p,q) = (1/n)√(Σ dist²(p[i], q_clean[i]))
        """
        sum_dist_sq = sum(
            self.manifold_single.dist(p[i], self.q_clean[i])**2 
            for i in range(self.n)
        )
        return np.sqrt(sum_dist_sq) / self.n
    
    def visualize(self, p_denoised=None, save_path=None):
        """Visualize clean, noisy, and optionally denoised signals"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Clean vs Noisy
        axes[0].plot(self.q_clean[:, 0], self.q_clean[:, 1], 'gray', 
                     linewidth=2, label='Clean signal', alpha=0.7)
        axes[0].scatter(self.q_noisy[:, 0], self.q_noisy[:, 1], c='teal', 
                        s=10, label='Noisy signal', alpha=0.6)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title('Clean and Noisy Signals on H²')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')
        
        # Plot 2: Comparison with denoised
        if p_denoised is not None:
            axes[1].scatter(self.q_noisy[:, 0], self.q_noisy[:, 1], c='teal', 
                            s=10, label='Noisy signal', alpha=0.4)
            axes[1].plot(p_denoised[:, 0], p_denoised[:, 1], 'cyan', 
                         linewidth=2, label='Denoised signal', alpha=0.8)
            axes[1].plot(self.q_clean[:, 0], self.q_clean[:, 1], 'gray', 
                         linewidth=1, label='Clean signal', alpha=0.5, 
                         linestyle='--')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
            axes[1].set_title('Noisy and Denoised Signals on H²')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_aspect('equal')
        else:
            axes[1].text(0.5, 0.5, 'Run optimization\nto see denoised signal', 
                         ha='center', va='center', transform=axes[1].transAxes,
                         fontsize=14)
            axes[1].set_title('Denoised Signal (pending)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def summary(self):
        """Print problem summary"""
        print("="*60)
        print("TV Denoising Problem on Hyperbolic Space")
        print("="*60)
        print(f"Signal dimension: {self.n}")
        print(f"TV parameter (α): {self.alpha}")
        print(f"Diameter (δ): {self.diameter:.4f}")
        print(f"Curvature bounds: ω = {self.omega}, Ω = {self.Omega}")
        print(f"Initial objective: {self.objective(self.initial_point()):.6f}")
        print("="*60)

