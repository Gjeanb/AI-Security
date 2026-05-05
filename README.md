# AI-Security
Trained a CNN on MNIST, attacked it with FGSM, and defended it with adversarial training.

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
