import argparse
import datetime
import os
from argparse import Namespace
from copy import deepcopy
from kan import *
import design_bench as db
import numpy as np
import torch
import torch.nn.functional as F
from design_bench.task import Task
from torch.optim import Adam

import wandb
from losses import get_loss_fn
from metrics import cal_overlap_auc, spearman_corr
from search import adam_search, grad_search,context_guided_search, two_stage_search
from utils import (
    build_data_loader,
    create_special_dataset_fast_unique,
    load_default_config,
    load_elite_data,
    record_from_dict,
    set_seed,
    task_kwargs,
)
from kan.MultKAN import MultKAN
_TKWARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


def run(args: Namespace):
    task_entry = f"MLP-{args.loss}"
    if args.expt_name:
        task_entry += f"-{args.expt_name}"

    set_seed(args.seed)

    task: Task = db.make(args.task, **task_kwargs(args.task))

    x = task.x.copy()
    y = task.y.copy()

    if args.eval_elites:
        x_elites, y_elites = load_elite_data(args.task, task)

    if args.normalize_ys:
        y = task.normalize_y(y)
        y_elites = task.normalize_y(y_elites)
    if task.is_discrete:
        x = task.to_logits(x)
        x_elites = task.to_logits(x_elites)
    if args.normalize_xs and (not task.is_discrete or args.normalize_logits):
        x = task.normalize_x(x)
        x_elites = task.normalize_x(x_elites)

    if args.use_wandb:
        run_name = f"{task_entry}-seed{args.seed}-{args.task}"
        ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
        ts_name = f"-ts-{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"

        wandb.login(key=args.wandb_api)

        wandb.init(
            project="Offline-Relation",
            name=run_name + ts_name,
            config=args.__dict__,
            group=f"{task_entry}",
            job_type=args.run_type,
            mode="online",
        )

    _shape0 = x.shape[1:]
    x = x.reshape(x.shape[0], -1)
    x_elites = x_elites.reshape(x_elites.shape[0], -1)

    x_elites = torch.from_numpy(x_elites).to(**_TKWARGS)
    y_elites = torch.from_numpy(y_elites).to(**_TKWARGS)

    x_train, y_train = create_special_dataset_fast_unique(
        x=x, y=y, m=args.list_length, num_samples=args.num_samples
    )

    train_loader, validate_loader = build_data_loader(
        x=x_train,
        y=y_train,
        batch_size=args.batch_size,
        require_valid=args.require_valid,
        valid_ratio_if_valid=args.valid_ratio,
        drop_last=args.drop_last,
    )

    iid_loader, _ = build_data_loader(
        x=x, y=y, batch_size=args.batch_size * 2, require_valid=False, drop_last=False
    )

    # forward_model = SimpleMLP(
    #     input_dim=x.shape[1], hidden_dim=args.hidden_dim, output_dim=args.output_dim
    # ).to(**_TKWARGS)

    forward_model = MultKAN(
        # width=[in_dim, [128, 0], [64, 0], [32, 0],[16, 0], out_dim],  # 更平滑的降维
        # width=[in_dim, [64, 0], [32, 0], [16, 0], out_dim],
        width=[x.shape[1], [32, 0], [16, 0], args.output_dim],  # 2025 10_22
        # width=[
        #     in_dim,
        #     [256, 6],
        #     [128, 4],
        #     [64, 3],
        #     out_dim
        # ],
        grid=3,  # 降低网格数量提高稳定性 # 3
        k=2,  # 降低多项式阶数
        device='cuda',
    )
    # forward_model = MultKAN(
    #     width=[60,32,1],
    #     grid=3,
    #     k=2,
    #     device="cuda",
    # )
    #forward_model = KAN.loadckpt("./my_model")
    loss_fn = get_loss_fn(args.loss, **args.loss_config)
    optimizer = Adam(
        params=forward_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    min_loss = float("inf")
    best_state_dict = None

    model_path = f"./model/MLP-{args.loss}-{args.task}-seed{args.seed}.pt"
    if not args.retrain_model and os.path.exists(model_path):
        model_path = f"{args.task}_{args.seed}"
        forward_model = KAN.loadckpt(f'./{model_path}')

    else:
        forward_model.fit(train_loader,validate_loader,iid_loader, x_elites, y_elites, args,opt="adam", steps=100, update_grid=False
                          , device='cuda',
                          temperature=args.contrastive_temperature)
        model_path = f"{args.task}_{args.seed}"
        forward_model.saveckpt(f"./{model_path}")
        # forward_model.loadckpt("./my_model")
        logger.info("load model  success!")
    if args.loss != "mse":
        pred_all = []
        for x_batch, _ in iid_loader:
            x_batch = x_batch.to(**_TKWARGS)
            y_pred = forward_model(x_batch)
            pred_all.append(y_pred)
        pred_all = torch.cat(pred_all, dim=0)
        pred_mean = pred_all.mean().item()
        pred_std = pred_all.std().item()
    else:
        pred_mean = 0.0
        pred_std = 1.0

    x_init = torch.Tensor(x[np.argsort(y.squeeze())[-args.num_solutions :]]).to(
        **_TKWARGS
    )

    x_res = deepcopy(x_init)

    if args.x_opt_method.lower() == "adam":
        x_res = adam_search(
            x_init=x_init,
            forward_model=forward_model,
            score_fn=lambda x: (loss_fn.score(x) - pred_mean) / pred_std,
            x_opt_lr=(
                args.x_opt_lr["discrete"]
                if task.is_discrete
                else args.x_opt_lr["continuous"]
            ),
            x_opt_step=(
                args.x_opt_step["discrete"]
                if task.is_discrete
                else args.x_opt_step["continuous"]
            ),
        )
    elif args.x_opt_method.lower() == "grad":
        x_res = grad_search(
            x_init=x_init,
            forward_model=forward_model,
            score_fn=lambda x: (loss_fn.score(x) - pred_mean) / pred_std,
            x_opt_lr=(
                args.x_opt_lr["discrete"]
                if task.is_discrete
                else args.x_opt_lr["continuous"]
            ),
            x_opt_step=(
                args.x_opt_step["discrete"]
                if task.is_discrete
                else args.x_opt_step["continuous"]
            ),
        )
    elif args.x_opt_method.lower() == "context":
        x_res = context_guided_search(
            x_init=x_init,
            forward_model=forward_model,
            score_fn=lambda out: (loss_fn.score(out) - pred_mean) / pred_std,
            x_opt_lr=(args.x_opt_lr["discrete"] if task.is_discrete else args.x_opt_lr["continuous"]),
            x_opt_step=(args.x_opt_step["discrete"] if task.is_discrete else args.x_opt_step["continuous"]),
            context_data=torch.from_numpy(x).to(**_TKWARGS),
            n_contexts=5,  # 可调: 每步采样的上下文数量
            context_size=10,  # 可调: 每个上下文列表的长度（含候选）
            max_radius=1.0  # 可调: 限制L2范围阈值
        )
    elif args.x_opt_method.lower() == "context2stage":
        # 这里的 score_fn 和前面保持一致
        score_fn = lambda out: (loss_fn.score(out) - pred_mean) / pred_std

        # 把原始训练 x 作为 context_data
        context_data_tensor = torch.from_numpy(x).to(**_TKWARGS)
        x_res, x_res_stage1, x_res_stage2 = two_stage_search(
            x_init=x_init,
            forward_model=forward_model,
            score_fn=score_fn,
            context_data=context_data_tensor,
            context_search_fn=context_guided_search,
            x_opt_lr_stage1=(
                args.x_opt_lr["discrete"]
                if task.is_discrete
                else args.x_opt_lr["continuous"]
            ),
            x_opt_step_stage1=(
                args.x_opt_step["discrete"]
                if task.is_discrete
                else args.x_opt_step["continuous"]
            ),
            n_contexts=args.n_contexts, # 3
            context_size=args.context_size, # 8
            max_radius=args.max_radius,  # 这一阶段保持稳
            topk_for_stage2=args.topk_for_stage2,  # 对 top-10 再冲一波
            x_opt_lr_stage2=(
                                args.x_opt_lr["discrete"]
                                if task.is_discrete
                                else args.x_opt_lr["continuous"]
                            ) * 1.5,
            x_opt_step_stage2=(
                                  args.x_opt_step["discrete"]
                                  if task.is_discrete
                                  else args.x_opt_step["continuous"]
                              ) * 2,
        )

    else:
        raise NotImplementedError("unknown search method")

    x_res = x_res.reshape((x_res.shape[0],) + tuple(_shape0)).detach().cpu().numpy()

    if args.normalize_xs:
        x_res = task.denormalize_x(x_res)
    if task.is_discrete:
        x_res = task.to_integers(x_res)

    forward_model.saveckpt("my_model")
    score = task.predict(x_res)
    score_100th = np.max(score)
    score_50th = np.median(score)
    score_25th = np.percentile(score, 25)
    score_75th = np.percentile(score, 75)

    dic2y = np.load("dic2y.npy", allow_pickle=True).item()
    y_min, y_max = dic2y[args.task]

    nmr_score_100th = (score_100th - y_min) / (y_max - y_min)
    nmr_score_75th = (score_75th - y_min) / (y_max - y_min)
    nmr_score_50th = (score_50th - y_min) / (y_max - y_min)
    nmr_score_25th = (score_25th - y_min) / (y_max - y_min)

    print(f"Score-100th: {nmr_score_100th}")
    print(f"Score-75th: {nmr_score_75th}")
    print(f"Score-50th: {nmr_score_50th}")
    print(f"Score-25th: {nmr_score_25th}")

    results_dict = {
        "Normalized-Score-100th": nmr_score_100th,
        "Normalized-Score-75th": nmr_score_75th,
        "Normalized-Score-50th": nmr_score_50th,
        "Normalized-Score-25th": nmr_score_25th,
        "Score-100th": score_100th,
        "Score-75th": score_75th,
        "Score-50th": score_50th,
        "Score-25th": score_25th,
        # "IID-MSE": iid_mse,
        # "IID-Rank-Correlation": iid_rank_corr,
        # "IID-AUC-PR": iid_auc_pr,
    }
    record_from_dict(
        metric_dict=results_dict, task=args.task, model=f"{task_entry}", seed=args.seed
    )

    if args.use_wandb:
        wandb.log(
            {
                "Normalized-Score/100th": nmr_score_100th,
                "Normalized-Score/75th": nmr_score_75th,
                "Normalized-Score/50th": nmr_score_50th,
                "Normalized-Score/25th": nmr_score_25th,
                "Score/100th": score_100th,
                "Score/75th": score_75th,
                "Score/50th": score_50th,
                "Score/25th": score_25th,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="TFBind10-Exact-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--loss",
        type=str,
        #default="rankcosine",
        default="rankcosine",
        choices=[
            "sigmoid_ce",
            "bce",
            "mse",
            "ranknet",
            "lambdarank",
            "rankcosine",
            "softmax",
            "listnet",
            "listmle",
            "approxndcg",
        ],
    )
    parser.add_argument("--list-length", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--wandb-api", type=str, default="5de2382d75ca4c87e8a090d9a1781d662651e6b2", help="WandB API key")
    parser.add_argument("--retrain-model", action="store_true", default=False)
    parser.add_argument("--expt-name", type=str, default="")
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-top-frac", type=float, default=0.3,
                        help="每个 list 内取前 top_frac 作为‘好样本’")
    parser.add_argument("--n_contexts", type=int, default=5)
    parser.add_argument("--context_size", type=int, default=12)
    parser.add_argument("--max_radius", type=float, default=1.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--topk_for_stage2", type=int, default=16)

    args = parser.parse_args()


    default_config = load_default_config(args.loss)
    default_config.update(args.__dict__)
    args.__dict__ = default_config

    os.makedirs("./model", exist_ok=True)

    run(args)
