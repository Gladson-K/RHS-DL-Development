import argparse
import torch
from torch import nn
from rhs_beamforming.data.synthetic import generate_debug_dataset
from rhs_beamforming.models import HaloBeamGNNNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--samples", type=int, default=20)
    args = p.parse_args()

    data = generate_debug_dataset(samples=args.samples)
    H = data["H"]
    target = data["targets"]["M"]

    model = HaloBeamGNNNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pred = model(H)["M"]
        loss = loss_fn(torch.view_as_real(pred), torch.view_as_real(target))
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch + 1} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
