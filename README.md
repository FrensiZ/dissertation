
# Token-Sequence Prediction

This repository contains code for generating token-based sequences using a combination of self-supervised learning and adversarial imitation learning techniques. The main model is based on the **RecurrentPPO** implementation from Stable Baselines3.

## Project Overview

The goal of this project is to simulate realistic token trajectories by training a generative model on observed real data. The self-supervised learning approach is used as a pre-training phase before applying adversarial imitation learning.

### Key Features

- **Self-Supervised Learning**: For pre-training the model on sequence data using teacher forcing.
- **Generative Adversarial Imitation Learning (GAIL)**: Imitating real token sequences by training a policy network.
- **RecurrentPPO**: Proximal Policy Optimization with recurrent networks for processing time-series data.

## Libraries Used

- **PyTorch**: For building the neural network models (LSTM, fully connected layers, etc.).
- **Stable Baselines3**: For implementing the RecurrentPPO algorithm and other reinforcement learning components.
- **SciPy & Statsmodels**: For time-series statistics and data analysis.
- **Gymnasium**: For creating the custom environment and handling the agent-environment interaction.
- **Matplotlib & Seaborn**: Used to visualize the results and model performance.

## Sections

### Data Preparation

The data preparation section simulates time series data using **Geometric Brownian Motion (GBM)**. GBM is commonly used to model stock prices and other financial variables. The formula used to generate the price sequence is:

$S_t = S_0 \exp\left(\left(\mu - rac{\sigma^2}{2}
ight)t + \sigma W_t
ight)$

Where:
- \(S_0\) is the initial price
- \(\mu\) is the drift (average return)
- \(\sigma\) is the volatility (standard deviation)
- \(W_t\) is a Wiener process (representing the random market component).

Example code for generating the data:

```python
import numpy as np
# Geometric Brownian Motion Parameters
mu = 0.1  # Drift
sigma = 0.2  # Volatility
S0 = 100  # Initial Price
T = 1  # Time horizon
N = 252  # Time steps

t = np.linspace(0, T, N)
W_t = np.random.standard_normal(size=N)
Price_GBM = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * np.cumsum(W_t))
```

### Model Setup

The core model uses a **Recurrent Neural Network (RNN)** architecture with **Long Short-Term Memory (LSTM)** cells. It’s set up as a policy network with shared layers between the actor (for policy learning) and the critic (for value function approximation). The main algorithm used is **RecurrentPPO**, which leverages the temporal structure of the data.

#### Model Architecture

- **Embedding Layer**: To convert input tokens into dense vectors.
- **LSTM Layer**: For sequential learning and processing token sequences.
- **Fully Connected Layers**: Output layers for the actor (policy) and critic (value function).
- **Discriminator**: A classifier that distinguishes real from generated sequences, trained using binary cross-entropy.

Example snippet from the model:

```python
class TokenLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TokenLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h)
        return out
```

### Training Process

The model is trained using a combination of **supervised learning** (for the pre-training phase) and **adversarial learning** (to fine-tune the policy). The discriminator is optimized with `BCEWithLogitsLoss`, while the actor-critic architecture is optimized with the **PPO** algorithm.

#### Optimization Details

- **Learning Rate**: 0.0003 (with scheduling for better convergence).
- **Batch Size**: 64 sequences per batch.
- **Loss Functions**:
    - Actor-Critic: **PPO loss**.
    - Discriminator: **Binary Cross-Entropy**.

### Results and Visualization

During training, the model is evaluated on its ability to predict future token sequences accurately. The notebook includes various plots like histograms and box plots to visualize the token distributions and model performance.

Example visualization:

```python
# Plot histogram of GBM prices
axs[0].hist(Price_GBM.flatten(), bins=20, color='blue', density=True)
axs[0].set_title('Histogram: GBM Prices')
axs[0].set_xlabel('Price')
axs[0].set_ylabel('Frequency')
```

### Custom Gym Environment

A custom **Gymnasium** environment is built to train the agent on token sequences. The environment includes:
- **Action Space**: Discrete action space for selecting tokens.
- **Observation Space**: Consisting of sequences of token embeddings.
- **Reward Function**: Rewarding the agent based on the similarity between generated and real token sequences.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Token-Sequence-Prediction.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Jupyter notebook to train the model:
```bash
jupyter notebook FINAL.ipynb
```

## Future Work

- **Hyperparameter Tuning**: Adjusting learning rates and PPO parameters for better performance.
- **Testing on Real Data**: Applying the model to real-world datasets to evaluate its practical performance.
- **Additional Evaluation Metrics**: Incorporating metrics like F1-score and AUC for a more comprehensive evaluation of the model’s performance.

---

This README provides a detailed explanation of the project’s workflow, from data preparation to training and model evaluation. Feel free to contribute or open an issue if you encounter any problems.
