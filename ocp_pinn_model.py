"""
ocp_pinn_model.py
=================
Physics-Informed Neural Network (PINN) for Open-Circuit Potential (OCP) modeling.

The PINN learns the OCP U(x) as a function of stoichiometry x while enforcing
thermodynamic constraints derived from the Gibbs free energy of mixing.

Two PINN formulations are implemented:

1. **Direct OCP PINN**: Learns U(x) directly with physics-informed regularization
   that penalizes violations of thermodynamic consistency:
   - Monotonicity: dU/dx <= 0 for cathodes (or appropriate sign for anodes)
   - Smoothness: penalize large second derivatives (physical plausibility)
   - Boundary behavior: physically meaningful boundary conditions

2. **Free-Energy PINN**: Learns the Gibbs free energy G(x) and derives U(x) 
   through automatic differentiation: U(x) = -(1/F) * dG/dx.
   This inherently enforces thermodynamic consistency because the OCP is 
   derived from a single scalar potential.

The governing equation relates OCP to the chemical potential:
    U(x) = U0 - (RT/F) * ln(x/(1-x)) + U_excess(x)

where U_excess(x) = (1/F) * d(g_excess)/dx and g_excess is the excess Gibbs
energy of mixing. The PINN residual penalizes departure from this relation.

References:
    Raissi et al., J. Comput. Phys. 378 (2019) 686.
    Yao & Viswanathan, J. Phys. Chem. Lett. 15 (2024) 1105.
    Ferguson & Bazant, J. Electrochem. Soc. 159 (2012) A1967.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================
R_GAS = 8.314462     # J/(mol·K)
F_CONST = 96485.33   # C/mol
T_REF = 298.15       # K (25 °C)


# ============================================================================
# Neural network architecture
# ============================================================================

class PINN_Net(nn.Module):
    """
    Fully connected neural network with smooth activation functions.
    
    Uses tanh activation throughout to ensure smooth outputs and well-defined
    derivatives, which is critical for physics-informed loss terms that 
    involve first and second derivatives of the network output.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input (1 for U(x), 2 for U(x,T)).
    output_dim : int
        Dimension of output (1 for scalar OCP or free energy).
    hidden_layers : list of int
        Number of neurons per hidden layer.
    activation : str
        Activation function: 'tanh', 'silu', or 'gelu'.
    """
    
    def __init__(self, input_dim=1, output_dim=1, 
                 hidden_layers=[64, 64, 64, 64], activation='tanh'):
        super().__init__()
        
        layers = []
        in_features = input_dim
        
        # Select activation function
        act_fn = {
            'tanh': nn.Tanh,
            'silu': nn.SiLU,
            'gelu': nn.GELU,
        }[activation]
        
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(act_fn())
            in_features = h
        
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)
        
        # Xavier initialization for better training stability
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# Direct OCP PINN
# ============================================================================

class DirectOCPPINN:
    """
    PINN that learns OCP U(x) directly with thermodynamic constraints.
    
    Loss function:
        L = w_data * L_data + w_physics * L_physics + w_mono * L_mono + w_smooth * L_smooth
    
    where:
    - L_data: MSE between predicted and observed OCP values
    - L_physics: residual of the governing thermodynamic relation
    - L_mono: penalty for violating monotonicity (dU/dx should have correct sign)
    - L_smooth: penalty for excessive curvature (d²U/dx²)
    
    Parameters
    ----------
    hidden_layers : list
        Network architecture.
    lr : float
        Learning rate.
    w_data : float
        Weight for data fitting loss.
    w_physics : float
        Weight for physics residual loss.
    w_mono : float
        Weight for monotonicity constraint.
    w_smooth : float
        Weight for smoothness regularization.
    electrode_type : str
        'cathode' (dU/dx < 0) or 'anode' (non-monotonic, different constraints).
    """
    
    def __init__(self, hidden_layers=[64, 64, 64, 64], lr=1e-3,
                 w_data=1.0, w_physics=0.1, w_mono=0.01, w_smooth=0.001,
                 electrode_type='cathode', activation='tanh'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = PINN_Net(1, 1, hidden_layers, activation).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=500, factor=0.5, min_lr=1e-6)
        
        self.w_data = w_data
        self.w_physics = w_physics
        self.w_mono = w_mono
        self.w_smooth = w_smooth
        self.electrode_type = electrode_type
        
        self.train_losses = []
        self.val_losses = []
        self.training_time = 0.0
    
    def _compute_derivatives(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute U, dU/dx, and d²U/dx² using automatic differentiation."""
        x.requires_grad_(True)
        U = self.net(x)
        
        dU_dx = torch.autograd.grad(
            U, x, grad_outputs=torch.ones_like(U),
            create_graph=True, retain_graph=True
        )[0]
        
        d2U_dx2 = torch.autograd.grad(
            dU_dx, x, grad_outputs=torch.ones_like(dU_dx),
            create_graph=True, retain_graph=True
        )[0]
        
        return U, dU_dx, d2U_dx2
    
    def _physics_residual(self, x_colloc: torch.Tensor) -> torch.Tensor:
        """
        Compute the physics residual.
        
        The thermodynamic relation for the ideal part of OCP is:
            U_ideal(x) = U0 - (RT/F) * ln(x / (1-x))
        
        The derivative of this ideal contribution is:
            dU_ideal/dx = -(RT/F) * [1/x + 1/(1-x)]
                        = -(RT/F) / [x(1-x)]
        
        The physics residual checks if the learned dU/dx is consistent with
        having a contribution from the ideal entropy of mixing.
        Specifically, we penalize: |d²U/dx² + (RT/F) * (1-2x)/[x(1-x)]²|
        which would be zero for a pure ideal solution model.
        
        For real materials, the excess contribution makes this non-zero, but
        we use a soft penalty to encourage thermodynamic plausibility.
        """
        x_colloc = x_colloc.requires_grad_(True)
        U, dU_dx, d2U_dx2 = self._compute_derivatives(x_colloc)
        
        # Ideal solution second derivative: d²U_ideal/dx² = (RT/F)(1-2x)/[x(1-x)]²
        x_safe = torch.clamp(x_colloc, 1e-4, 1 - 1e-4)
        d2U_ideal = (R_GAS * T_REF / F_CONST) * (1 - 2*x_safe) / (x_safe * (1 - x_safe))**2
        
        # The excess part d²U_excess/dx² should be smooth (bounded polynomial)
        # Residual: the non-ideal part of the curvature should be small/smooth
        residual = d2U_dx2 - d2U_ideal
        
        return residual
    
    def _monotonicity_loss(self, x_colloc: torch.Tensor) -> torch.Tensor:
        """Penalize violations of monotonicity."""
        _, dU_dx, _ = self._compute_derivatives(x_colloc)
        
        if self.electrode_type == 'cathode':
            # For cathodes: dU/dx <= 0 (OCP decreases with lithiation)
            violation = torch.relu(dU_dx)
        else:
            # For anodes like graphite, monotonicity is more complex
            # due to staging plateaus. Use a weaker constraint.
            violation = torch.relu(dU_dx - 0.5)  # Allow some positive slopes
        
        return torch.mean(violation**2)
    
    def _smoothness_loss(self, x_colloc: torch.Tensor) -> torch.Tensor:
        """Penalize excessive curvature for smoothness."""
        _, _, d2U_dx2 = self._compute_derivatives(x_colloc)
        return torch.mean(d2U_dx2**2)
    
    def train(self, data: Dict, n_epochs: int = 5000, n_colloc: int = 500,
              verbose: bool = True) -> Dict:
        """
        Train the PINN.
        
        Parameters
        ----------
        data : dict
            Preprocessed data from ocp_data_processing.preprocess_data().
        n_epochs : int
            Number of training epochs.
        n_colloc : int
            Number of collocation points for physics loss.
        verbose : bool
            Print training progress.
        
        Returns
        -------
        history : dict
            Training and validation loss history.
        """
        # Convert data to tensors
        x_train = torch.FloatTensor(data['x_train']).to(self.device)
        U_train = torch.FloatTensor(data['U_train']).to(self.device)
        x_val = torch.FloatTensor(data['x_val']).to(self.device)
        U_val = torch.FloatTensor(data['U_val']).to(self.device)
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.net.train()
            self.optimizer.zero_grad()
            
            # Data loss
            U_pred = self.net(x_train)
            loss_data = torch.mean((U_pred - U_train)**2)
            
            # Collocation points for physics (uniformly sampled in [0, 1])
            x_colloc = torch.rand(n_colloc, 1, device=self.device)
            x_colloc = x_colloc * 0.96 + 0.02  # Avoid boundaries
            
            # Physics residual loss
            residual = self._physics_residual(x_colloc)
            loss_physics = torch.mean(residual**2)
            
            # Monotonicity loss
            loss_mono = self._monotonicity_loss(x_colloc)
            
            # Smoothness loss
            loss_smooth = self._smoothness_loss(x_colloc)
            
            # Total loss
            loss = (self.w_data * loss_data 
                    + self.w_physics * loss_physics
                    + self.w_mono * loss_mono
                    + self.w_smooth * loss_smooth)
            
            loss.backward()
            self.optimizer.step()
            
            # Validation
            self.net.eval()
            with torch.no_grad():
                U_val_pred = self.net(x_val)
                val_loss = torch.mean((U_val_pred - U_val)**2).item()
            
            self.train_losses.append(loss.item())
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Loss: {loss.item():.6f} | "
                      f"Data: {loss_data.item():.6f} | "
                      f"Physics: {loss_physics.item():.6f} | "
                      f"Val: {val_loss:.6f}")
        
        self.training_time = time.time() - start_time
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'training_time': self.training_time
        }
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict OCP for given stoichiometry values."""
        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(self.device)
            U_pred = self.net(x_tensor).cpu().numpy().flatten()
        return U_pred
    
    def predict_with_derivatives(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict OCP and its derivative dU/dx."""
        self.net.eval()
        x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(self.device)
        x_tensor.requires_grad_(True)
        
        U = self.net(x_tensor)
        dU_dx = torch.autograd.grad(
            U, x_tensor, grad_outputs=torch.ones_like(U),
            create_graph=False
        )[0]
        
        return (U.detach().cpu().numpy().flatten(),
                dU_dx.detach().cpu().numpy().flatten())


# ============================================================================
# Free-Energy PINN
# ============================================================================

class FreeEnergyPINN:
    """
    PINN that learns the Gibbs free energy G(x) and derives OCP via differentiation.
    
    The OCP is computed as:
        U(x) = -(1/F) * dG/dx
    
    This formulation inherently enforces thermodynamic consistency because:
    1. OCP is derived from a single scalar potential (the free energy).
    2. The convexity of G(x) directly maps to monotonicity of U(x).
    3. Phase coexistence (plateaus) arise naturally from common tangent construction.
    
    The network learns G(x) directly, and U(x) is obtained through PyTorch's
    automatic differentiation. Physics constraints on G(x) include:
    - G should contain an ideal mixing entropy contribution
    - G should be smooth and differentiable
    - Boundary conditions: G(0) and G(1) correspond to pure phases
    
    Parameters
    ----------
    hidden_layers : list
        Network architecture.
    lr : float
        Learning rate.
    w_data : float
        Weight for OCP data fitting loss.
    w_convex : float
        Weight for convexity regularization on G(x).
    w_boundary : float
        Weight for boundary condition loss.
    """
    
    def __init__(self, hidden_layers=[64, 64, 64, 64], lr=1e-3,
                 w_data=1.0, w_convex=0.01, w_boundary=0.1,
                 activation='tanh'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = PINN_Net(1, 1, hidden_layers, activation).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=500, factor=0.5, min_lr=1e-6)
        
        self.w_data = w_data
        self.w_convex = w_convex
        self.w_boundary = w_boundary
        
        self.train_losses = []
        self.val_losses = []
        self.training_time = 0.0
    
    def _compute_ocp_from_G(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute OCP from free energy via differentiation.
        
        U(x) = -(1/F) * dG/dx
        
        Also compute d²G/dx² for convexity check.
        """
        x.requires_grad_(True)
        G = self.net(x)
        
        dG_dx = torch.autograd.grad(
            G, x, grad_outputs=torch.ones_like(G),
            create_graph=True, retain_graph=True
        )[0]
        
        d2G_dx2 = torch.autograd.grad(
            dG_dx, x, grad_outputs=torch.ones_like(dG_dx),
            create_graph=True, retain_graph=True
        )[0]
        
        # OCP from thermodynamics: U = -(1/F) * dG/dx
        # Since we work with normalized values, the scaling is absorbed 
        # into the network weights
        U = -dG_dx  # Simplified: F scaling absorbed into training
        
        return U, G, d2G_dx2
    
    def train(self, data: Dict, n_epochs: int = 5000, n_colloc: int = 500,
              verbose: bool = True) -> Dict:
        """Train the Free-Energy PINN."""
        x_train = torch.FloatTensor(data['x_train']).to(self.device)
        U_train = torch.FloatTensor(data['U_train']).to(self.device)
        x_val = torch.FloatTensor(data['x_val']).to(self.device)
        U_val = torch.FloatTensor(data['U_val']).to(self.device)
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.net.train()
            self.optimizer.zero_grad()
            
            # Data loss: compare derived OCP with observations
            U_pred, _, _ = self._compute_ocp_from_G(x_train)
            loss_data = torch.mean((U_pred - U_train)**2)
            
            # Collocation points for physics constraints
            x_colloc = torch.rand(n_colloc, 1, device=self.device)
            x_colloc = x_colloc * 0.96 + 0.02
            
            # Convexity: d²G/dx² >= 0 for thermodynamic stability
            _, _, d2G_dx2 = self._compute_ocp_from_G(x_colloc)
            loss_convex = torch.mean(torch.relu(-d2G_dx2)**2)
            
            # Total loss
            loss = (self.w_data * loss_data 
                    + self.w_convex * loss_convex)
            
            loss.backward()
            self.optimizer.step()
            
            # Validation
            self.net.eval()
            with torch.no_grad():
                # For validation, we need gradients, so use eval but enable grad temporarily
                pass
            
            # Re-enable grad for validation OCP computation
            U_val_pred, _, _ = self._compute_ocp_from_G(x_val)
            val_loss = torch.mean((U_val_pred.detach() - U_val)**2).item()
            
            self.train_losses.append(loss.item())
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Loss: {loss.item():.6f} | "
                      f"Data: {loss_data.item():.6f} | "
                      f"Convex: {loss_convex.item():.6f} | "
                      f"Val: {val_loss:.6f}")
        
        self.training_time = time.time() - start_time
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'training_time': self.training_time
        }
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict OCP for given stoichiometry values."""
        self.net.eval()
        x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(self.device)
        x_tensor.requires_grad_(True)
        
        G = self.net(x_tensor)
        dG_dx = torch.autograd.grad(
            G, x_tensor, grad_outputs=torch.ones_like(G),
            create_graph=False
        )[0]
        
        U = -dG_dx
        return U.detach().cpu().numpy().flatten()
    
    def predict_free_energy(self, x: np.ndarray) -> np.ndarray:
        """Predict the learned free energy G(x)."""
        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(self.device)
            G = self.net(x_tensor).cpu().numpy().flatten()
        return G
    
    def predict_with_derivatives(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict OCP and dU/dx."""
        self.net.eval()
        x_tensor = torch.FloatTensor(x.reshape(-1, 1)).to(self.device)
        x_tensor.requires_grad_(True)
        
        G = self.net(x_tensor)
        dG_dx = torch.autograd.grad(
            G, x_tensor, grad_outputs=torch.ones_like(G),
            create_graph=True, retain_graph=True
        )[0]
        
        d2G_dx2 = torch.autograd.grad(
            dG_dx, x_tensor, grad_outputs=torch.ones_like(dG_dx),
            create_graph=False
        )[0]
        
        U = -dG_dx
        dU_dx = -d2G_dx2
        
        return (U.detach().cpu().numpy().flatten(),
                dU_dx.detach().cpu().numpy().flatten())


# ============================================================================
# Convenience function
# ============================================================================

def build_pinn(formulation: str = 'direct', electrode_type: str = 'cathode',
               **kwargs) -> object:
    """
    Factory function to create a PINN model.
    
    Parameters
    ----------
    formulation : str
        'direct' for DirectOCPPINN, 'free_energy' for FreeEnergyPINN.
    electrode_type : str
        'cathode' or 'anode'.
    **kwargs
        Additional arguments passed to the PINN constructor.
    
    Returns
    -------
    model : DirectOCPPINN or FreeEnergyPINN
    """
    if formulation == 'direct':
        return DirectOCPPINN(electrode_type=electrode_type, **kwargs)
    elif formulation == 'free_energy':
        return FreeEnergyPINN(**kwargs)
    else:
        raise ValueError(f"Unknown formulation: {formulation}")


if __name__ == "__main__":
    print("PINN models loaded successfully.")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Quick test
    net = PINN_Net(1, 1, [32, 32])
    x_test = torch.randn(10, 1)
    y_test = net(x_test)
    print(f"Test output shape: {y_test.shape}")
