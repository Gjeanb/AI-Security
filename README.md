# AI-Security
This project shows how deep learning models can be tricked by adversarial inputs and how adversarial training can restore robustness. A CNN was trained on the MNIST handwritten digit dataset, attacked using the Fast Gradient Sign Method (FGSM) and then hardened through adversarial training.

## Results
| | Clean | Under Attack (ε=0.25) |
|---|---|---|
| Baseline | 98.27% | 2.77% |
| Defended | 95.96% | 94.08% |

## How to Run
pip install torch torchvision matplotlib
py ai_security_capstone.py

## Tools
Python · PyTorch · Matplotlib
