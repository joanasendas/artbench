# Project: ArtBench Generative Modeling

This project focuses on implementing and training various generative models (Autoencoders, GANs, and Diffusion Models) using the ArtBench-10 dataset.

## Project Structure

- `ArtBench-10/`: Contains the local dataset files in both binary and Python formats, along with a metadata CSV.
- `scripts/`:
    - `artbench_local_dataset.py`: Core utility for loading the ArtBench-10 dataset into PyTorch-compatible formats.
- `student_start_pack/`:
    - `1 - Autoencoders and Variational Autoencoders.ipynb`: Implementation of AE and VAE models.
    - `2 - Generative Adversarial Network.ipynb/py`: Implementation of DCGAN and related GAN architectures.
    - `5 - Diffusion Models and Latent Diffusion Models.ipynb/py`: Implementation of Pixel-space and Latent Diffusion Models.
    - `runs/`: Directory where training outputs, checkpoints (`.pt`, `.pth`), and metrics are stored.
    - `exported_data/`: Contains subsets of data used for specific training runs.
- `genai-env/`: Python virtual environment for the project.

## Technologies Used

- **Framework:** PyTorch
- **Metrics:** `torchmetrics` (FID, KID)
- **Hyperparameter Tuning:** Optuna
- **Visualization:** Matplotlib, PIL
- **Dataset Handling:** Hugging Face `datasets` (via custom loading scripts)

## Getting Started

### Environment Setup
1. Ensure the `genai-env` environment is activated.
2. Install dependencies (if not already present):
   ```bash
   pip install tqdm torchmetrics optuna pandas requests torchvision
   ```

### Data Loading
The project uses a local copy of ArtBench-10. The `build_loaders` function in the scripts handles the creation of training and testing DataLoaders. It defaults to a limited subset for quick experimentation but can be configured for the full 50k image set.

### Training and Evaluation
- Training scripts save checkpoints to `student_start_pack/runs/`.
- The "best" models are typically saved based on the lowest FID score (`best_fid_model.pt`).
- Evaluation uses `run_robust_evaluation` to compute FID and KID over multiple runs for statistical reliability.

## Key Conventions

- **Device Selection:** Scripts automatically use CUDA if available, falling back to CPU.
- **Reproducibility:** A `set_seed(42)` function is used throughout to ensure consistent results.
- **Checkpointing:** Both regular epoch checkpoints and "best FID" checkpoints are maintained.
- **EMA:** Exponential Moving Average (EMA) of model weights is used in Diffusion models to improve sample quality.

## Usage Notes

- When running scripts/notebooks, ensure the working directory is `student_start_pack/` to maintain correct relative path references to the dataset and output directories.
- If running from the project root, paths like `runs/diffusion/...` may need to be prefixed with `student_start_pack/`.
