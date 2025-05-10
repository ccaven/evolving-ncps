"""
Generates a synthetic dataset of a linear dynamical system.
"""

import torch

from typing_extensions import Self
from dataclasses import dataclass

@dataclass
class DynamicalSystemParameters:
    timesteps: int
    input_dim: int
    hidden_dim: int
    observation_dim: int

    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor

    device: str

    def infinite_input_batches(
        params: Self,
        batch_size: int
    ):
        timesteps = params.timesteps
        input_dim = params.input_dim
        
        while True:
            # generate linear progressions
            x = torch.linspace(0, torch.pi * 2, timesteps, device=params.device) \
                .unsqueeze(0) \
                .unsqueeze(-1) \
                .repeat(batch_size, 1, input_dim)
            
            # frequency shift
            x *= torch.rand((batch_size, 1, input_dim), device=params.device) * 4 + 1

            # phase shift
            x += torch.rand((batch_size, 1, input_dim), device=params.device) * torch.pi * 2

            yield torch.sin(x)

    def infinite_lds_batches(
        params: Self,
        batch_size: int
    ):
        x = torch.zeros((batch_size, params.timesteps, params.hidden_dim), device=params.device)
        y = torch.zeros((batch_size, params.timesteps, params.observation_dim), device=params.device)
        w = torch.randn(batch_size, params.timesteps, params.hidden_dim, device=params.device) * 0.1
        v = torch.randn(batch_size, params.timesteps, params.observation_dim, device=params.device) * 0.01

        for u in params.infinite_input_batches(batch_size):
            x[:, 0].normal_()
            for t in range(params.timesteps):
                if t < params.timesteps - 1:
                    x[:, t + 1] = \
                        x[:, t] @ params.A.T + \
                        u[:, t] @ params.B.T + w[:, t]
                y[:, t] = x[:, t] @ params.C.T + u[:, t] @ params.D.T + v[:, t]
            yield u, x, y

def main():
    test_lds = DynamicalSystemParameters(
        timesteps=64,
        input_dim=2,
        hidden_dim=4,
        observation_dim=1,
        A=torch.tensor([
            [0.8, -0.3, 0, 0],
            [0.3, 0.8, 0, 0],
            [0, 0, 0.7, -0.4],
            [0, 0, 0.4, 0.7]
        ], device="cuda"),
        B=torch.randn(4, 2, device="cuda"),
        C=torch.randn(1, 4, device="cuda"),
        D=torch.randn(1, 2, device="cuda"),
        device="cuda"
    )

    batch_size = 128

    batch_generator = test_lds.infinite_lds_batches(batch_size)

    for filename, num_elements in (
        ("lds_train.pkl", 4096),
        ("lds_valid.pkl", 512),
        ("lds_test.pkl", 256)
    ):
        all_batches = [[item.cpu().detach() for item in next(batch_generator)] for _ in range(num_elements // batch_size)]
        all_us = torch.cat([bundle[0] for bundle in all_batches])
        all_xs = torch.cat([bundle[1] for bundle in all_batches])
        all_ys = torch.cat([bundle[2] for bundle in all_batches])

        torch.save({
            "u": all_us,
            "x": all_xs,
            "y": all_ys
        }, filename)

if __name__ == "__main__":
    main()