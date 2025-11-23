from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from copy import deepcopy
def adam_search(
    x_init: torch.Tensor,
    forward_model: nn.Module,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    x_opt_lr: float = 1e-3,
    x_opt_step: int = 100,
) -> torch.Tensor:

    x_res = x_init.clone()

    for i in tqdm(range(len(x_init)), desc="Searching x with Adam"):
        x_i = x_init[i : i + 1].clone()
        x_i.requires_grad = True
        x_opt = torch.optim.Adam(params=[x_i], lr=x_opt_lr)
        opt_step = x_opt_step
        for _ in range(opt_step):
            x_opt.zero_grad()
            y_pred = forward_model(x_i)
            score = -score_fn(y_pred)
            score.backward()
            x_opt.step()

        with torch.no_grad():
            x_res[i] = x_i.data

    return x_res


def grad_search(
    x_init: torch.Tensor,
    forward_model: nn.Module,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    x_opt_lr: float = 1e-3,
    x_opt_step: int = 100,
) -> torch.Tensor:

    x_res = x_init.clone()

    for i in tqdm(range(len(x_init)), desc="Searching x with Grad"):
        x_i = x_init[i : i + 1].clone()
        x_i.requires_grad = True
        opt_step = x_opt_step

        for _ in range(opt_step):
            y_pred = forward_model(x_i)
            score = -score_fn(y_pred)
            _grad = torch.autograd.grad(outputs=score, inputs=x_i)[0]
            x_i = x_i + x_opt_lr * _grad

        with torch.no_grad():
            x_res[i] = x_i.data

    return x_res

import torch
import torch.nn as nn
from typing import Callable




def context_guided_search(
        x_init: torch.Tensor,
        forward_model: torch.nn.Module,
        score_fn,
        x_opt_lr: float,
        x_opt_step: int,
        context_data: torch.Tensor,
        n_contexts: int = 5,
        context_size: int = 10,
        max_radius: float = 1.0,
) -> torch.Tensor:
    """
    基于代理模型的 Context-Guided 梯度搜索。
    Args:
        x_init: 初始候选集张量，形状 (batch_size, input_dim)
        forward_model: 训练好的代理模型 (KAN surrogate)
        score_fn: 得分函数，应用于 forward_model 输出上，例如 lambda out: (loss_fn.score(out) - mean) / std
        x_opt_lr: 基础学习率（将在优化过程中按照 curriculum 调整）
        x_opt_step: 优化迭代步数
        context_data: 用于构造列表上下文的参考数据集张量
        n_contexts: 每次迭代为每个候选采样的列表上下文数量
        context_size: 每个列表上下文的总样本数（含候选本身）
        max_radius: 限制候选与初始值的最大 L2 距离阈值
    Returns:
        优化后的候选集张量，形状与 x_init 相同
    """
    # 确保 context_data 在与模型相同的设备上
    device = next(forward_model.parameters()).device
    context_data = context_data.to(device)
    # 初始化可训练候选变量
    x = x_init.clone().detach().to(device)
    x.requires_grad_(True)
    # 使用 Adam 优化器进行梯度上升（通过最小化负目标实现）
    optimizer = torch.optim.Adam([x], lr=x_opt_lr)

    for step in tqdm(range(x_opt_step)):
        # **Curriculum 学习率调度**: 随迭代步数线性增加学习率（早期0.1倍，后期逐步增至1.0倍）
        lr_scale = 0.1 + 0.9 * (step / (x_opt_step - 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = x_opt_lr * lr_scale

        # 正向传播计算目标得分（多个上下文平均排名分数）
        optimizer.zero_grad()
        total_score = 0.0  # 将对所有候选取平均，为方便梯度累加也可用 sum 后再平均
        for i in range(x.shape[0]):
            candidate = x[i:i + 1]  # 当前候选 (1, input_dim)
            # 通过代理模型计算候选的预测分数
            cand_pred = forward_model(candidate)
            cand_score = score_fn(cand_pred)  # 标量张量 (形状 [1])，表示候选的预测得分
            # 确保 cand_score 为标量
            cand_score_val = cand_score.squeeze()

            # 计算候选在多个随机上下文中的平均胜出概率（排名得分）
            context_rank_scores = []
            for _ in range(n_contexts):
                # 从 context_data 随机采样 context_size-1 个其他样本组成上下文
                # 注意避免放入与候选完全相同的样本（此处假定上下文样本来自训练集，与候选可能略有差异）
                idx = np.random.choice(len(context_data), size=context_size - 1, replace=False)
                context_samples = context_data[idx]  # 张量形状 (context_size-1, input_dim)
                # 将候选与上下文样本一起通过模型计算预测
                all_samples = torch.cat([candidate, context_samples], dim=0)  # (context_size, input_dim)
                preds = forward_model(all_samples)  # 形状 (context_size, 1)
                scores = score_fn(preds).squeeze()  # 转换为1维张量，长度 context_size
                # 提取候选和其他的得分
                cand_s = scores[0]  # 候选样本的得分
                others_s = scores[1:]  # 其他样本的得分 (长度 context_size-1)
                # 计算候选胜出概率：sigmoid(cand - others) 表示候选比分数高的概率
                win_probs = torch.sigmoid(cand_s - others_s)
                # 胜出概率均值作为该上下文下候选的排名分数
                context_rank_scores.append(win_probs.mean())

            # 该候选在所有上下文下的平均排名得分
            cand_avg_rank_score = torch.stack(context_rank_scores).mean()
            # 累加到总得分（针对batch内多个候选取均值，这里先求和稍后除以数量）
            total_score += cand_avg_rank_score

        # 求平均目标得分
        avg_score = total_score / x.shape[0]
        # 梯度上升：最大化 avg_score 等价于最小化 -avg_score
        loss = -avg_score
        loss.backward()
        optimizer.step()

        # **半径约束：**将更新后的候选限制在距初始样本的L2半径内
        with torch.no_grad():
            diff = x - x_init.to(device)
            # 计算每个候选的 L2 距离
            diff_norm = torch.norm(diff.view(diff.size(0), -1), dim=1)
            # 若超过阈值则拉回边界
            for j in range(x.shape[0]):
                if diff_norm[j] > max_radius:
                    x[j] = x_init[j].to(device) + (diff[j] * (max_radius / diff_norm[j]))

    return x

def two_stage_search(
    x_init: torch.Tensor,
    forward_model: torch.nn.Module,
    score_fn,
    context_data: torch.Tensor,
    # 第一阶段：context 搜索（稳）
    context_search_fn,          # 你已有的 context_guided_search
    x_opt_lr_stage1: float,
    x_opt_step_stage1: int,
    n_contexts: int = 5,
    context_size: int = 10,
    max_radius: float = 1.0,
    # 第二阶段：原始 adam_search（冲 100th）
    adam_search_fn=None,        # 你已有的 adam_search
    topk_for_stage2: int = 10,
    x_opt_lr_stage2: float = None,
    x_opt_step_stage2: int = None,
):
    """
    Stage1: context_search_fn 提升整体分布 (50th/75th)
    Stage2: 对 Stage1 的 top-k 再用 adam_search_fn 冲极值 (100th)
    """
    device = next(forward_model.parameters()).device
    x_init = x_init.to(device)
    context_data = context_data.to(device)

    # ===== Stage 1: 稳，context 搜索 =====
    x_stage1 = context_search_fn(
        x_init=x_init,
        forward_model=forward_model,
        score_fn=score_fn,
        x_opt_lr=x_opt_lr_stage1,
        x_opt_step=x_opt_step_stage1,
        context_data=context_data,
        n_contexts=n_contexts,
        context_size=context_size,
        max_radius=max_radius,
    )   # 形状 [B1, D]

    # surrogate 打分
    with torch.no_grad():
        pred1 = forward_model(x_stage1)
        scores1 = score_fn(pred1).view(-1)   # [B1]

    # ===== 选出 top-k 做第二阶段起点 =====
    k = min(topk_for_stage2, x_stage1.shape[0])
    topk_idx = torch.topk(scores1, k=k).indices
    x_init_stage2 = x_stage1[topk_idx].detach().clone()      # [k, D]

    # ===== Stage 2: 冲，原始 adam_search =====

    if x_opt_lr_stage2 is None:
        x_opt_lr_stage2 = x_opt_lr_stage1 * 1.5  # 稍微激进一点
    if x_opt_step_stage2 is None:
        x_opt_step_stage2 = x_opt_step_stage1 * 2

    x_stage2 = adam_search(
        x_init=deepcopy(x_init_stage2),
        forward_model=forward_model,
        score_fn=score_fn,
        x_opt_lr=x_opt_lr_stage2,
        x_opt_step=x_opt_step_stage2,
    )   # [k, D]

    # ===== 合并两阶段候选 =====
    x_final = torch.cat([x_stage1, x_stage2], dim=0)  # [B1+k, D]

    return x_final, x_stage1, x_stage2

