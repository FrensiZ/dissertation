# GAIL within LOBSTER
## Current Workflow
### Data Preparation
- Create a 1D Autoregressive simulator
- Preprocess data to create state-action pairs
### Pre-train the S5 Model:
- Pre-train the policy network (S5 model) on the data (supervised learing). policy $\pi$
### GAIL Setup
- Initialize policy network $\pi$
- Initiality discriminator $\phi$
### Training Loop:
- **Generate Data**: Use $\pi$ to generate trajectories
- **Train Discriminator**: Train $\phi$ to distuingish between $\pi_E$ and $\pi$.
- **Policy Update**: Update $\pi_E$ using RL signal with $\phi$ output as reward.
### Evaluation and Refinement:
- Evaluate / Benchmark $\pi$ and $\pi_E$ with "LOB-Bench: Towards Benchmarking Generative AI for Finance"
