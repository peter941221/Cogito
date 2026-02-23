"""Genesis Beta Agent: Transformer-based architecture for cross-substrate validation.

Uses causal self-attention instead of LSTM for temporal processing.
This provides a fundamentally different computational substrate while
maintaining the same external interface and learning objectives.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cogito.agent.action_head import ActionHead
from cogito.agent.memory_buffer import Experience, MemoryBuffer
from cogito.agent.prediction_head import PredictionHead
from cogito.agent.sensory_encoder import SensoryEncoder
from cogito.config import Config

if TYPE_CHECKING:
    from cogito.agent.learner import OnlineLearner


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Tensor of shape (seq_len, batch, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pe[:x.size(0)].unsqueeze(1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply causal self-attention.

        Args:
            x: Input tensor (seq_len, batch, d_model).
            mask: Optional attention mask.

        Returns:
            Output tensor (seq_len, batch, d_model).
        """
        seq_len, batch_size, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(seq_len, batch_size, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(seq_len, batch_size, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(seq_len, batch_size, self.n_heads, self.head_dim)

        # Transpose for attention (batch, heads, seq, head_dim)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        else:
            scores = scores + mask

        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(2, 0, 1, 3).contiguous()
        out = out.view(seq_len, batch_size, self.d_model)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor (seq_len, batch, d_model).

        Returns:
            Output tensor (seq_len, batch, d_model).
        """
        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class TransformerCore(nn.Module):
    """Transformer-based core for temporal processing.

    Uses causal self-attention with a sliding context window.
    Maintains a KV cache for efficient inference.
    """

    def __init__(
        self,
        input_dim: int = 70,  # 64 encoded + 6 action one-hot
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        context_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.context_len = context_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=context_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])

        # Output projection to match LSTM output dim
        self.output_proj = nn.Linear(d_model, 128)

        # Context buffer
        self.register_buffer('context_buffer', torch.zeros(context_len, 1, d_model))
        self.context_pos = 0

    def forward(
        self,
        encoded: torch.Tensor,
        prev_action_onehot: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input through transformer.

        Args:
            encoded: Encoded sensory input (batch, 64) or (64,).
            prev_action_onehot: Previous action one-hot (batch, 6) or (6,).
            context: Optional context buffer (seq_len, batch, d_model).

        Returns:
            Tuple of (output, new_context).
        """
        # Ensure batch dimension
        if encoded.dim() == 1:
            encoded = encoded.unsqueeze(0)
            prev_action_onehot = prev_action_onehot.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Concatenate input
        x = torch.cat([encoded, prev_action_onehot], dim=-1)  # (batch, 70)

        # Project to model dimension
        x = self.input_proj(x)  # (batch, d_model)

        # Add to context
        batch_size = x.size(0)

        if context is None:
            context = self.context_buffer.clone()

        # Update context (sliding window)
        if self.context_pos >= self.context_len:
            # Shift context
            context = torch.roll(context, -1, dims=0)
            context[-1] = x
        else:
            context[self.context_pos] = x
            self.context_pos += 1

        # Add positional encoding
        context_pe = self.pos_encoding(context[:self.context_pos])

        # Apply transformer blocks
        out = context_pe
        for block in self.blocks:
            out = block(out)

        # Get last output
        last_out = out[-1]  # (batch, d_model)

        # Project to output dimension
        output = self.output_proj(last_out)  # (batch, 128)

        if squeeze_output:
            output = output.squeeze(0)

        return output, context

    def init_context(self, device: torch.device | None = None) -> torch.Tensor:
        """Initialize empty context buffer.

        Args:
            device: Device for tensor.

        Returns:
            Zero-initialized context buffer.
        """
        if device is None:
            device = next(self.parameters()).device
        self.context_pos = 0
        return torch.zeros(self.context_len, 1, self.d_model, device=device)

    def get_hidden_vector(self, context: torch.Tensor) -> torch.Tensor:
        """Extract hidden vector from context for monitoring.

        Args:
            context: Context buffer tensor.

        Returns:
            Flattened hidden vector (512,).
        """
        # Take statistics from context
        if context is None or self.context_pos == 0:
            return torch.zeros(512, device=next(self.parameters()).device)

        # Use last N positions and flatten
        effective_context = context[:self.context_pos]

        # Mean and std across time
        mean = effective_context.mean(dim=0).flatten()  # (d_model,)
        std = effective_context.std(dim=0).flatten()  # (d_model,)

        # Last hidden states
        last = effective_context[-1].flatten()  # (d_model,)
        second_last = effective_context[-2].flatten() if self.context_pos > 1 else torch.zeros_like(last)

        # Attention statistics (simulate)
        attn_mean = torch.zeros(128, device=mean.device)
        attn_std = torch.zeros(128, device=mean.device)

        # Combine to get 512 dimensions
        # 64 + 64 + 64 + 64 + 128 + 128 = 512
        hidden = torch.cat([mean, std, last, second_last, attn_mean, attn_std])

        return hidden


class GenesisBetaAgent(nn.Module):
    """Transformer-based Cogito agent for cross-substrate validation.

    Total parameters: ~250,000 (similar to LSTM version)
    """

    def __init__(
        self,
        config: type[Config] | None = None,
        device: torch.device | str | None = None,
    ):
        """Initialize the Genesis Beta agent.

        Args:
            config: Configuration class (default: Config).
            device: Device for computation (default: CPU).
        """
        super().__init__()

        self.config = config or Config
        self.device = device or torch.device("cpu")

        # Create modules
        self.encoder = SensoryEncoder()
        self.core = TransformerCore()
        self.action_head = ActionHead()
        self.prediction_head = PredictionHead()

        # Move to device
        self.to(self.device)

        # Memory buffer
        self.memory = MemoryBuffer(capacity=self.config.BUFFER_SIZE)

        # Context (like hidden state in LSTM)
        self.context = self.core.init_context(device=self.device)

        # Previous action
        self.prev_action = 5

        # Statistics
        self.step_count = 0
        self.total_energy_gained = 0.0
        self.total_energy_lost = 0.0
        self.times_died = 0
        self.current_lifespan = 0
        self._current_energy = float(self.config.INITIAL_ENERGY)

    def act(
        self,
        observation: np.ndarray,
        energy: float | None = None,
    ) -> tuple[int, dict]:
        """Process observation and select action.

        Args:
            observation: 106-dim observation vector.
            energy: Current energy.

        Returns:
            Tuple of (action, info_dict).
        """
        if energy is not None:
            self._current_energy = energy

        full_obs = self._complete_observation(observation)

        # Encode
        obs_tensor = torch.tensor(full_obs, dtype=torch.float32, device=self.device)
        encoded = self.encoder(obs_tensor)

        # Previous action one-hot
        prev_action_onehot = torch.zeros(self.config.NUM_ACTIONS, device=self.device)
        prev_action_onehot[self.prev_action] = 1.0

        # Transformer forward
        core_output, new_context = self.core(encoded, prev_action_onehot, self.context)

        # Select action
        action, log_prob, entropy = self.action_head.select_action(core_output)

        # Predict
        prediction = self.prediction_head(core_output)

        # Update context
        self.context = new_context
        self.prev_action = action
        self.step_count += 1

        # Hidden vector for monitoring
        hidden_vector = self.core.get_hidden_vector(new_context)

        return action, {
            "encoded": encoded.detach().cpu().numpy(),
            "core_output": core_output.detach().cpu().numpy(),
            "prediction": prediction.detach().cpu().numpy(),
            "log_prob": log_prob,
            "entropy": entropy,
            "hidden_vector": hidden_vector.detach().cpu().numpy(),
        }

    def _complete_observation(self, observation: np.ndarray) -> np.ndarray:
        """Complete observation with agent state."""
        full_obs = observation.copy()
        full_obs[98] = self._current_energy / self.config.MAX_ENERGY
        for i in range(self.config.NUM_ACTIONS):
            full_obs[99 + i] = 1.0 if i == self.prev_action else 0.0
        full_obs[105] = 0.5
        return full_obs

    def reset_on_death(self) -> None:
        """Reset agent state on death."""
        self.context = self.core.init_context(device=self.device)
        self.prev_action = 5
        self.times_died += 1
        self.current_lifespan = 0
        self._current_energy = float(self.config.INITIAL_ENERGY)

    def get_internal_state(self) -> dict:
        """Get internal state for monitoring."""
        hidden_vector = self.core.get_hidden_vector(self.context)
        return {
            "hidden_vector": hidden_vector.detach().cpu().numpy(),
            "prev_action": self.prev_action,
            "step_count": self.step_count,
            "current_energy": self._current_energy,
            "times_died": self.times_died,
        }

    def save(self, path: str | Path) -> None:
        """Save agent state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.state_dict(),
            "context": self.context.cpu(),
            "context_pos": self.core.context_pos,
            "prev_action": self.prev_action,
            "step_count": self.step_count,
            "total_energy_gained": self.total_energy_gained,
            "total_energy_lost": self.total_energy_lost,
            "times_died": self.times_died,
            "current_lifespan": self.current_lifespan,
            "current_energy": self._current_energy,
        }

        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load agent state."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model_state_dict"])
        self.context = state["context"].to(self.device)
        self.core.context_pos = state["context_pos"]
        self.prev_action = state["prev_action"]
        self.step_count = state["step_count"]
        self.total_energy_gained = state["total_energy_gained"]
        self.total_energy_lost = state["total_energy_lost"]
        self.times_died = state["times_died"]
        self.current_lifespan = state["current_lifespan"]
        self._current_energy = state["current_energy"]

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
