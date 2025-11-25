import torch
import argparse
import logging
import os, sys
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils

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

# wukong specific parameters
parser.add_argument("--num_layers", type=int, default=3, help="number of interaction layers in Wukong")
parser.add_argument("--num_embed_lcb", type=int, default=16, help="number of embedding in Linear Compress Block in Wukong")
parser.add_argument(
    "--num_embed_fmb", type=int, default=16, help="number of embedding in Factorization Machine Block in Wukong"
)
parser.add_argument("--rank_fmb", type=int, default=24, help="rank in Wukong")
parser.add_argument("--fm_mlp_dims", type=int, nargs='+', default=[512, 512], help="sub-mlp within the FM Block in Wukong")

# Rankmixer specific parameters
parser.add_argument("--num_L", type=int, default=3, help="number of layers in the model")
parser.add_argument("--expansion_rate", type=int, default=4, help="expansion rate for the model")

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


class Tester(object):

    def __init__(self, opt):
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])
        self.network = trainUtils.getModel(opt["model"], opt["model_opt"]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = trainUtils.getOptim(self.network, opt["optimizer"], self.lr, self.l2)
        # self.network.load_state_dict(torch.load(self.model_dir, map_location=self.device))
        # self.network.eval()

    def eval_on_batch(self, data):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def evaluate(self, on: str):
        preds, trues = [], []
        run_index = 0
        for feature, label in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
            # run_index += 1
            # if run_index % 10 == 0:
            #     break
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss, y_pred, y_true

    def eval_tts_on_batch(self, data, index):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network.forward_with_mask(data, index)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def evaluate_tts(self, on: str, index=0):
        preds, trues = [], []
        run_index = 0
        for feature, label in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred = self.eval_tts_on_batch(feature, index)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
            # run_index += 1
            # if run_index % 10 == 0:
            #     break
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss, y_pred, y_true

    def evaluate_tts_random_on_batch(self, data):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network.forward_with_random_mask(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def evaluate_tts_random(self, on: str):
        preds, trues = [], []
        run_index = 0
        for feature, label in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred = self.evaluate_tts_random_on_batch(feature)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
            # run_index += 1
            # if run_index % 10 == 0:
            #     break
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss, y_pred, y_true

    def evaluate_tts_random_dropout_on_batch(self, data):
        self.network.train()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network.forward_with_random_mask(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def evaluate_tts_random_dropout(self, on: str):
        preds, trues = [], []
        run_index = 0
        for feature, label in self.dataloader.get_data(on, batch_size=self.bs):
            pred = self.evaluate_tts_random_dropout_on_batch(feature)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
            # run_index += 1
            # if run_index % 10 == 0:
            #     break
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss, y_pred, y_true

    def test1_seeds(self, seed, opt):
        # mydir = 'save/' + opt['dataset'].lower() + '-' + opt['model'].lower() + '/feature-' + str(seed) + '/'
        mydir = os.path.dirname(self.model_dir) + '/feature-' + str(seed)
        print(mydir)
        os.makedirs(mydir, exist_ok=True)
        # self.network.load_state_dict(torch.load('save/' + opt['dataset'].lower() + '-' + opt['model'].lower() + '-'+str(seed)+'.pt', map_location=self.device))
        self.network.load_state_dict(torch.load(self.model_dir, map_location=self.device))
        self.network.eval()
        self.test1(savedir=mydir)

    def test1(self, savedir=""):
        print("Testing...")
        test_auc, test_loss, full_pred, full_true = self.evaluate("test")
        print(f"Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}")

        y_trues, y_preds = [], []
        for index in range(args.field):
            test_auc, test_loss, y_pred, y_true = self.evaluate_tts("test", index=index)
            print(f"Index: {index:.1f}, Test TTS AUC: {test_auc:.6f}, Test TTS Loss: {test_loss:.6f}")
            y_preds.append(np.expand_dims(y_pred, axis=0))
            y_trues.append(np.expand_dims(y_true, axis=0))

        tts_y_pred = np.concatenate(y_preds, axis=0)
        tts_y_true = np.concatenate(y_trues, axis=0)

        print(tts_y_pred.shape, tts_y_true.shape)

        np.save(savedir + "/full_y_pred.npy", full_pred)
        np.save(savedir + "/full_y_true.npy", full_true)
        np.save(savedir + "/tts_y_pred.npy", tts_y_pred)

        # ===== #

        # tts_y_pred = np.load("tts_y_pred.npy")
        # full_true = np.load("full_y_true.npy")
        # full_pred = np.load("full_y_pred.npy")

        # confident = ((full_pred > 0.5) == full_true)
        # full_correct = np.argwhere(confident == True)
        # full_incorrect = np.argwhere(confident == False)
        # print(np.sum(confident), confident.shape)
        # print(full_correct.shape, full_incorrect.shape)
        # print(full_correct[:10])

        # auc = metrics.roc_auc_score(full_true, full_pred)
        # loss = metrics.log_loss(full_true, full_pred)
        # print(f"Original AUC: {auc:.6f}, Original Loss: {loss:.6f}")

        # i = 101
        # print(tts_y_pred[:,i,0], full_pred[i,0], full_true[i,0])

        # i = 100
        # print(tts_y_pred[:,i,0], full_pred[i,0], full_true[i,0])

        # max_y_pred = np.max(tts_y_pred, axis=0)
        # min_y_pred = np.min(tts_y_pred, axis=0)
        # mean_y_pred = np.mean(tts_y_pred, axis=0)
        # std_y_pred = np.std(tts_y_pred, axis=0)

        # correct_std = std_y_pred[full_correct[:, 0], full_correct[:, 1]]
        # incorrect_std = std_y_pred[full_incorrect[:, 0], full_incorrect[:, 1]]

        # print(np.mean(correct_std), np.std(correct_std))
        # print(np.mean(incorrect_std), np.std(incorrect_std))

        # confident = np.where(mean_y_pred > 0.25, max_y_pred, min_y_pred)
        # tts_final = max_y_pred

        # # confident = ((mean_y_pred > 0.5) == full_true)
        # # print(np.sum(confident), confident.shape)

        # auc = metrics.roc_auc_score(full_true, tts_final)
        # loss = metrics.log_loss(full_true, tts_final)
        # print(f"Final TTS AUC: {auc:.6f}, Final TTS Loss: {loss:.6f}")

    def test2_seeds(self, seed, opt):
        # mydir = 'save/' + opt['dataset'].lower() + '-' + opt['model'].lower() + '/noise-' + str(seed) + '/'
        mydir = os.path.dirname(self.model_dir) + '/noise-' + str(seed)
        os.makedirs(mydir, exist_ok=True)
        # self.network.load_state_dict(torch.load('save/' + opt['dataset'].lower() + '-' + opt['model'].lower() + '-'+str(seed)+'.pt', map_location=self.device))
        self.network.load_state_dict(torch.load(self.model_dir, map_location=self.device))
        self.network.eval()
        self.test2(savedir=mydir)

    def test2(self, savedir=""):
        print("Testing...")
        test_auc, test_loss, full_pred, full_true = self.evaluate("test")
        print(f"Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}")

        y_trues, y_preds = [], []
        for index in range(64):
            test_auc, test_loss, y_pred, y_true = self.evaluate_tts_random("test")
            print(f"Index: {index:.1f}, Test TTS AUC: {test_auc:.6f}, Test TTS Loss: {test_loss:.6f}")
            y_preds.append(np.expand_dims(y_pred, axis=0))
            y_trues.append(np.expand_dims(y_true, axis=0))

        tts_y_pred = np.concatenate(y_preds, axis=0)
        tts_y_true = np.concatenate(y_trues, axis=0)

        print(tts_y_pred.shape, tts_y_true.shape)

        np.save(savedir + "/full_y_pred.npy", full_pred)
        np.save(savedir + "/full_y_true.npy", full_true)
        np.save(savedir + "/tts_y_pred.npy", tts_y_pred)

        # ===== #

        # tts_y_pred = np.load("tts_y_pred.npy")
        # full_true = np.load("full_y_true.npy")
        # full_pred = np.load("full_y_pred.npy")

        # confident = ((full_pred > 0.5) == full_true)
        # full_correct = np.argwhere(confident == True)
        # full_incorrect = np.argwhere(confident == False)

        # max_y_pred = np.max(tts_y_pred, axis=0)
        # min_y_pred = np.min(tts_y_pred, axis=0)
        # mean_y_pred = np.mean(tts_y_pred, axis=0)
        # std_y_pred = np.std(tts_y_pred, axis=0)

        # max_y_pred = np.max(tts_y_pred, axis=0)
        # min_y_pred = np.min(tts_y_pred, axis=0)
        # mean_y_pred = np.mean(tts_y_pred, axis=0)
        # std_y_pred = np.std(tts_y_pred, axis=0)

        # tts_final = max_y_pred

        # correct_std = std_y_pred[full_correct[:, 0], full_correct[:, 1]]
        # incorrect_std = std_y_pred[full_incorrect[:, 0], full_incorrect[:, 1]]

        # print(np.mean(correct_std), np.std(correct_std))
        # print(np.mean(incorrect_std), np.std(incorrect_std))

        # auc = metrics.roc_auc_score(full_true, tts_final)
        # loss = metrics.log_loss(full_true, tts_final)
        # print(f"Final TTS AUC: {auc:.6f}, Final TTS Loss: {loss:.6f}")

    def test3_seeds(self, seed, opt):
        # mydir = 'save/' + opt['dataset'].lower() + '-' + opt['model'].lower() + '/random-' + str(seed) + '/'
        mydir = os.path.dirname(self.model_dir) + '/dropout-' + str(seed)
        os.makedirs(mydir, exist_ok=True)
        # self.network.load_state_dict(torch.load('save/' + opt['dataset'].lower() + '-' + opt['model'].lower() + '-'+str(seed)+'.pt', map_location=self.device))
        self.network.load_state_dict(torch.load(self.model_dir, map_location=self.device))
        self.network.train()
        self.test3(savedir=mydir)

    def test3(self, savedir=""):
        print("Testing...")
        test_auc, test_loss, full_pred, full_true = self.evaluate("test")
        print(f"Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}")

        y_trues, y_preds = [], []
        for index in range(64):
            test_auc, test_loss, y_pred, y_true = self.evaluate_tts_random_dropout("test")
            print(f"Index: {index:.1f}, Test TTS AUC: {test_auc:.6f}, Test TTS Loss: {test_loss:.6f}")
            y_preds.append(np.expand_dims(y_pred, axis=0))
            y_trues.append(np.expand_dims(y_true, axis=0))

        tts_y_pred = np.concatenate(y_preds, axis=0)
        tts_y_true = np.concatenate(y_trues, axis=0)

        print(tts_y_pred.shape, tts_y_true.shape)

        np.save(savedir + "/full_y_pred.npy", full_pred)
        np.save(savedir + "/full_y_true.npy", full_true)
        np.save(savedir + "/tts_y_pred.npy", tts_y_pred)

    def test0_seeds(self, seed, opt):
        # mydir = 'save/' + opt['dataset'].lower() + '-' + opt['model'].lower() + '/dropout-' + str(seed) + '/'
        mydir = os.path.dirname(self.model_dir) + '/naive-' + str(seed)

        os.makedirs(mydir, exist_ok=True)
        # self.network.load_state_dict(torch.load('save/' + opt['dataset'].lower() + '-' + opt['model'].lower() + '-'+str(seed)+'.pt', map_location=self.device))
        self.network.load_state_dict(torch.load(self.model_dir, map_location=self.device))
        self.network.train()
        self.test0(savedir=mydir)

    def test0(self, savedir=""):
        print("Testing...")
        self.log_dir = str(self.model_dir)[:-3] + ".log"
        test_auc, test_loss, full_pred, full_true = self.evaluate("test")
        with open(self.log_dir, 'a') as f:
            print(f"Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}", file=f)

        np.save(savedir + "/full_y_pred.npy", full_pred)
        np.save(savedir + "/full_y_true.npy", full_true)


def main():

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
        # Rankmixer specific parameters
        "num_L": args.num_L,
        "expansion_rate": args.expansion_rate
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
    tester = Tester(opt)
    tester.test0_seeds(args.seed, opt)
    # tester.test1_seeds(args.seed, opt)
    # tester.test2_seeds(args.seed, opt)
    # tester.test3_seeds(args.seed, opt)


if __name__ == "__main__":

    main()
