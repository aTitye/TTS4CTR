import torch
import argparse
import logging
import os, sys
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils
import modules.models as models
import time

parser = argparse.ArgumentParser(description="optfs trainer")
parser.add_argument("dataset", type=str, help="specify dataset")
parser.add_argument("model", type=str, help="specify model")

# dataset information
parser.add_argument("--feature", type=int, help="feature number", required=True)
parser.add_argument("--field", type=int, help="field number", required=True)
parser.add_argument("--data_dir", type=str, help="data directory", required=True)

# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate", default=3e-5)
parser.add_argument("--l2", type=float, help="L2 regularization", default=1e-3)
parser.add_argument("--bsize", type=int, help="batchsize", default=4096)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--max_epoch", type=int, default=20, help="maxmium epochs")
parser.add_argument("--save_dir", type=Path, help="model save directory")

# neural network hyperparameters
parser.add_argument("--dim", type=int, help="embedding dimension", default=16)
parser.add_argument("--mlp_dims", type=int, nargs='+', default=[1024, 512, 256], help="mlp layer size")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="mlp dropout rate (default:0.0)")
parser.add_argument("--mlp_bn", action="store_true", help="mlp batch normalization")
parser.add_argument("--cross", type=int, help="cross layer", default=3)
parser.add_argument(
    "--model_dim",
    action="append",
    default=[],
    help="override latent dim per sub-model as model=dim",
)

# wukong specific parameters
parser.add_argument("--num_layers", type=int, default=3, help="number of interaction layers in Wukong")
parser.add_argument("--num_embed_lcb", type=int, default=16, help="number of embedding in Linear Compress Block in Wukong")
parser.add_argument(
    "--num_embed_fmb", type=int, default=16, help="number of embedding in Factorization Machine Block in Wukong"
)
parser.add_argument("--rank_fmb", type=int, default=24, help="rank in Wukong")
parser.add_argument("--fm_mlp_dims", type=int, nargs='+', default=[512, 512], help="sub-mlp within the FM Block in Wukong")

# Rankmixer specific parameters
parser.add_argument("--num_L", type=int, default=3, help="number of layers in RankMixer")
parser.add_argument("--expansion_rate", type=int, default=2, help="expansion rate in RankMixer")

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=-1, help="device info")
parser.add_argument("--seed", type=int, default=2021, help="random seed")

args = parser.parse_args()

my_seed = args.seed
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'


class Trainer(object):

    def __init__(self, opt):
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])
        modellist = opt["model"].split("+")
        print("Training group model:", modellist)
        print("Model options:", opt["model_opt"])
        self.network = models.GroupModel(opt["model_opt"], modellist).to(self.device)
        # print(self.network)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = trainUtils.getOptim(self.network, opt["optimizer"], self.lr, self.l2)

    def train_on_batch(self, label, data):
        self.network.train()
        self.optim.zero_grad()
        data, label = data.to(self.device), label.to(self.device)
        logit = self.network(data)
        #print(logit.shape)
        logloss = self.criterion(logit, label)
        regloss = self.network.reg()
        # print(logloss, regloss)
        loss = regloss + logloss
        loss.backward()
        self.optim.step()
        return logloss.item()

    def eval_on_batch(self, data):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def train(self, epochs):
        step = 0
        cur_auc = 0.0
        early_stop = False
        for epoch_idx in range(int(epochs)):
            epoch_start = time.time()
            train_loss = .0
            step = 0
            for feature, label in self.dataloader.get_data("train", batch_size=self.bs):
                train_loss += self.train_on_batch(label, feature)
                step += 1
            train_loss /= step
            epoch_train_time = time.time() - epoch_start
            val_auc, val_loss, val_infer_time = self.evaluate("val")
            print(
                "[Epoch {epoch:d} | Train Loss:{loss:.6f} | Train Time:{ttrain:.3f}s | Val AUC:{val_auc:.6f}, Val Loss:{val_loss:.6f} | Val Inf Time:{vtime:.3f}s]"
                .format(
                    epoch=epoch_idx,
                    loss=train_loss,
                    ttrain=epoch_train_time,
                    val_auc=val_auc,
                    val_loss=val_loss,
                    vtime=val_infer_time
                )
            )
            if val_auc > cur_auc:
                cur_auc = val_auc
                torch.save(self.network.state_dict(), self.model_dir)
            else:
                self.network.load_state_dict(torch.load(self.model_dir))
                self.network.to(self.device)
                early_stop = True
                te_auc, te_loss, te_infer_time = self.evaluate("test")

                self.log_dir = str(self.model_dir)[:-3] + ".log"
                print(
                    "Early stop at epoch {epoch:d}|Test AUC: {te_auc:.6f}, Test Loss:{te_loss:.6f} | Test Inf Time:{tinf:.3f}s"
                    .format(epoch=epoch_idx, te_auc=te_auc, te_loss=te_loss, tinf=te_infer_time)
                )
                with open(self.log_dir, 'a') as f:
                    print(f"Test AUC: {te_auc:.6f}, Test Loss: {te_loss:.6f}, Test Inf Time: {te_infer_time:.3f}s", file=f)
                break
        if not early_stop:
            te_auc, te_loss, te_infer_time = self.evaluate("test")
            print(
                "Final Test AUC:{te_auc:.6f}, Test Loss:{te_loss:.6f}, Test Inf Time:{tinf:.3f}s".format(
                    te_auc=te_auc, te_loss=te_loss, tinf=te_infer_time
                )
            )
            self.log_dir = str(self.model_dir)[:-3] + ".log"
            with open(self.log_dir, 'a') as f:
                print(f"Test AUC: {te_auc:.6f}, Test Loss: {te_loss:.6f}, Test Inf Time: {te_infer_time:.3f}s", file=f)

    def evaluate(self, on: str):
        preds, trues = [], []
        infer_time = 0.0
        for feature, label in self.dataloader.get_data(on, batch_size=self.bs):
            t0 = time.time()
            pred = self.eval_on_batch(feature)
            infer_time += time.time() - t0
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss, infer_time


def main():
    model_dim_overrides = {}
    for override in args.model_dim:
        try:
            name, value = override.split("=", 1)
            name = name.strip().lower()
            value = int(value.strip())
        except ValueError:
            raise ValueError(f"Invalid --model_dim entry '{override}'. Use format model=dim")
        model_dim_overrides[name] = value

    model_opt = {
        "latent_dim": args.dim,
        "feat_num": args.feature,
        "field_num": args.field,
        "mlp_dropout": args.mlp_dropout,
        "use_bn": args.mlp_bn,
        "mlp_dims": args.mlp_dims,
        # DCN specific parameters
        "cross": args.cross,
        # Wukong specific parameters
        "num_layers": args.num_layers,
        "num_embed_lcb": args.num_embed_lcb,
        "num_embed_fmb": args.num_embed_fmb,
        "rank_fmb": args.rank_fmb,
        "fm_mlp_dims": args.fm_mlp_dims,
        # RankMixer specific parameters
        "num_L": args.num_L,
        "expansion_rate": args.expansion_rate,
        "model_dims": model_dim_overrides,
    }
    opt = {
        "model_opt": model_opt,
        "dataset": args.dataset,
        "model": args.model,
        "lr": args.lr,
        "l2": args.l2,
        "bsize": args.bsize,
        "epoch": args.max_epoch,
        "optimizer": args.optim,
        "data_dir": args.data_dir,
        "save_dir": args.save_dir,
        "cuda": args.cuda
    }
    print(opt)
    trainer = Trainer(opt)
    trainer.train(args.max_epoch)
    # trainer.calibrate(args.max_epoch)


if __name__ == "__main__":
    """
    python trainer_group.py Criteo
    """
    main()
