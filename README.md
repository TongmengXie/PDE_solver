# PDE Solver using Physics-Informed Neural Networks

This repository demonstrates the implementation of Physics-Informed Neural Networks (PINNs) for solving Partial Differential Equations (PDEs). PINNs combine the power of neural networks with physical constraints to solve differential equations without requiring labeled training data.

## Implemented PDE Solutions

The repository includes two different PDE solvers:

### 1. First-Order PDE Solver
Solves the PDE: `du/dx = 2du/dt + u`
- Boundary condition: `u(x,0) = 6e^(-3x)`
- Domain: `x ∈ [0,2], t ∈ [0,1]`
- Solution available in `solve_PDE_NN.ipynb`
- Trained model saved as `model_uxt.pt`

### 2. Diffusion Equation Solver
Solves the PDE: `d²u/dx² = du/dt`
- Boundary condition: `u(x,0) = sin(πx)`
- Domain: `x ∈ [0,1], t ∈ [0,0.5]`
- Solution available in `solve_PDE_NN_diffusion.ipynb`
- Trained model saved as `model_uxt_diffusion.pt`

## Project Structure

- `solve_PDE_NN.ipynb`: Implementation of the first-order PDE solver
- `solve_PDE_NN_diffusion.ipynb`: Implementation of the diffusion equation solver
- `model_uxt.pt`: Saved PyTorch model for the first-order PDE solution
- `model_uxt_diffusion.pt`: Saved PyTorch model for the diffusion equation solution
- `final_sample3000_ver0_2.csv`, `view1.csv`: Sample data files

## Neural Network Architecture

Both implementations use a similar neural network architecture:
- Input layer: 2 nodes (x, t coordinates)
- 5 hidden layers with 5 neurons each
- Output layer: 1 node (solution u(x,t))
- Activation function: Sigmoid for hidden layers, linear for output layer
- Loss function: MSE (Mean Squared Error)
- Optimizer: Adam

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- mpl_toolkits (for 3D visualization)

The code automatically utilizes CUDA if available (`device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`).

## Usage

1. Open the desired Jupyter notebook:
   - `solve_PDE_NN.ipynb` for the first-order PDE
   - `solve_PDE_NN_diffusion.ipynb` for the diffusion equation

2. Key parameters that can be modified:
   - Number of training iterations (default: 20000)
   - Number of sample points for boundary conditions (default: 500)
   - Neural network architecture (layers and neurons)
   - Domain ranges for x and t

3. The training process combines two loss terms:
   - Boundary condition loss
   - PDE residual loss

4. Results are visualized using 3D surface plots showing the solution u(x,t) over the domain.

## Visualization

The solutions are visualized using 3D surface plots that show:
- x-axis: Spatial dimension
- y-axis: Time dimension
- z-axis: Solution value u(x,t)
- Color mapping: Solution magnitude

## Model Saving

Trained models are automatically saved to:
- `model_uxt.pt` for the first-order PDE
- `model_uxt_diffusion.pt` for the diffusion equation

These can be loaded later for inference or continued training.

## License

This project is licensed under the terms of the LICENSE file included in the repository.
