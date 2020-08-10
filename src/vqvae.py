import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tnrange, tqdm
import optuna
import os
import random

from models import VQVAE, BinaryFileSource, LBG
from loss_func import calc_lf0_rmse, vqvae_loss
from util import create_loader, train, test, parse

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_vqvae(args, trial=None):
    """
    """
    model = VQVAE(
        num_layers=args["num_layers"], z_dim=args["z_dim"], num_class=args["num_class"]
    ).to(device)

    train_loader, test_loader = create_loader()
    train_num = int(args["train_ratio"] * len(train_loader))  # 1
    train_loader = train_loader[:train_num]

    if args["model_path"] != "":
        model.load_state_dict(torch.load(args["model_path"]))

    else:
        lbg = LBG(num_class=args["num_class"], z_dim=args["z_dim"])
        # zを用意

        sampled_indices = random.sample(
            list(range(len(train_loader))), min(len(train_loader), 100)
        )
        z = torch.tensor([[0.0] * args["z_dim"]]).to(device)

        print("コードブックを初期化")
        for index in tqdm(sampled_indices):
            data = train_loader[index]
            with torch.no_grad():
                z_tmp = model.encode(
                    torch.tensor(data[0]).to(device),
                    torch.tensor(data[1]).to(device),
                    data[2],
                ).view(-1, args["z_dim"])
                z = torch.cat([z, z_tmp], dim=0).to(device)
        init_codebook = torch.from_numpy(lbg.calc_q_vec(z)).to(device)
        codebook = nn.Parameter(init_codebook)
        model.init_codebook(codebook)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)  # 1e-3
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    loss_list = []
    test_loss_list = []
    f0_loss_list = []
    start = time.time()
    for epoch in range(1, args["num_epoch"] + 1):
        loss = train(epoch, model, train_loader, vqvae_loss, optimizer)
        test_loss, f0_loss = test(epoch, model, test_loader, vqvae_loss)

        print(
            "epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}".format(
                epoch + 1, args["num_epoch"], loss, test_loss
            )
        )
        # scheduler.step()
        # logging
        loss_list.append(loss)
        test_loss_list.append(test_loss)
        f0_loss_list.append(f0_loss)

        if trial is not None:
            trial.report(test_loss, epoch - 1)

        if trial is not None:
            if trial.should_prune():
                return optuna.TrialPruned()

        print(time.time() - start)

        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                args["output_dir"] + "/vqvae_model_{}.pth".format(epoch),
            )
        np.save(args["output_dir"] + "/loss_list.npy", np.array(loss_list))
        np.save(args["output_dir"] + "/test_loss_list.npy", np.array(test_loss_list))
        np.save(args["output_dir"] + "/test_f0loss_list.npy", np.array(f0_loss_list))

    return f0_loss


if __name__ == "__main__":
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)
    train_vqvae(vars(args))

