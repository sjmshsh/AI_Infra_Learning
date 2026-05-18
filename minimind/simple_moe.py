import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    一个专家就是一个小型前馈神经网络 FFN。

    输入形状:
        [N, hidden_dim]

    输出形状:
        [N, hidden_dim]
    """

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleMoE(nn.Module):
    """
    一个最小版 MoE 层。

    核心流程:
    1. Router 给每个 token 分配专家概率
    2. 每个 token 只选择 top-k 个专家
    3. 被选中的专家处理对应 token
    4. 专家输出乘以 router 权重
    5. 加权结果累加回原 token 位置
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        ffn_dim: int = 32,
    ):
        super().__init__()

        assert top_k <= num_experts, "top_k 必须小于等于 num_experts"

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Router / Gate
        # 输入:  [hidden_dim]
        # 输出:  [num_experts]
        #
        # 它的作用是:
        # 对每个 token 计算它应该去每个专家的分数。
        self.router = nn.Linear(hidden_dim, num_experts)

        # 创建多个专家
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x, return_router_info: bool = False):
        """
        x 的形状:
            [batch_size, seq_len, hidden_dim]

        例如:
            batch_size = 2
            seq_len = 3
            hidden_dim = 8

        那么一共有:
            2 * 3 = 6 个 token
        """

        batch_size, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim

        # --------------------------------------------------
        # 1. 展平 token
        # --------------------------------------------------
        # 原来:
        #   x: [batch_size, seq_len, hidden_dim]
        #
        # 变成:
        #   x_flat: [num_tokens, hidden_dim]
        #
        # 这样更方便让每个 token 独立选择专家。
        num_tokens = batch_size * seq_len
        x_flat = x.reshape(num_tokens, hidden_dim)

        # --------------------------------------------------
        # 2. Router 计算每个 token 对每个专家的分数
        # --------------------------------------------------
        # router_logits:
        #   [num_tokens, num_experts]
        #
        # 例如 num_tokens = 6, num_experts = 4:
        #   router_logits.shape = [6, 4]
        #
        # 每一行表示一个 token 对 4 个专家的打分。
        router_logits = self.router(x_flat)

        # --------------------------------------------------
        # 3. softmax 得到专家概率
        # --------------------------------------------------
        # router_probs:
        #   [num_tokens, num_experts]
        #
        # 每一行加起来等于 1。
        router_probs = F.softmax(router_logits, dim=-1)

        # --------------------------------------------------
        # 4. 每个 token 选择 top-k 个专家
        # --------------------------------------------------
        # topk_probs:
        #   [num_tokens, top_k]
        #
        # topk_indices:
        #   [num_tokens, top_k]
        #
        # 举例:
        #   topk_indices[0] = [2, 0]
        #
        # 表示第 0 个 token 选择了:
        #   专家 2 和 专家 0
        topk_probs, topk_indices = torch.topk(
            router_probs,
            self.top_k,
            dim=-1,
        )

        # --------------------------------------------------
        # 5. 对 top-k 权重重新归一化
        # --------------------------------------------------
        # 假设某个 token 的 top-2 概率是:
        #   [0.40, 0.30]
        #
        # 它们加起来是 0.70。
        #
        # 重新归一化后:
        #   [0.40 / 0.70, 0.30 / 0.70]
        #   = [0.5714, 0.4286]
        #
        # 这样被选中的专家权重加起来就是 1。
        topk_weights = topk_probs / topk_probs.sum(
            dim=-1,
            keepdim=True,
        )

        # --------------------------------------------------
        # 6. 准备输出容器
        # --------------------------------------------------
        # output_flat:
        #   [num_tokens, hidden_dim]
        #
        # 后面每个专家的结果会累加到这里。
        output_flat = torch.zeros_like(x_flat)

        # --------------------------------------------------
        # 7. 每个专家只处理分配给自己的 token
        # --------------------------------------------------
        for expert_id, expert in enumerate(self.experts):

            # mask:
            #   [num_tokens, top_k]
            #
            # mask[n, k] = True 表示:
            #   第 n 个 token 的第 k 个选择是当前 expert_id
            #
            # 例如:
            #   topk_indices[n] = [2, 0]
            #
            # 如果 expert_id = 2:
            #   mask[n] = [True, False]
            #
            # 如果 expert_id = 0:
            #   mask[n] = [False, True]
            mask = (topk_indices == expert_id)

            # 如果没有任何 token 选择这个专家，就跳过
            if not mask.any():
                continue

            # token_positions:
            #   哪些 token 选择了当前专家
            #
            # selected_slots:
            #   当前专家在这些 token 的 top-k 里排第几
            #
            # 举例:
            #   token_positions = [0, 3, 5]
            #   selected_slots   = [1, 0, 1]
            #
            # 表示:
            #   token 0 的 top-k 第 1 个位置选中了当前专家
            #   token 3 的 top-k 第 0 个位置选中了当前专家
            #   token 5 的 top-k 第 1 个位置选中了当前专家
            token_positions, selected_slots = mask.nonzero(as_tuple=True)

            # 取出当前专家要处理的 token
            #
            # expert_input:
            #   [当前专家接收的 token 数量, hidden_dim]
            expert_input = x_flat[token_positions]

            # 当前专家进行计算
            #
            # expert_output:
            #   [当前专家接收的 token 数量, hidden_dim]
            expert_output = expert(expert_input)

            # 取出对应的 router 权重
            #
            # weights:
            #   [当前专家接收的 token 数量, 1]
            weights = topk_weights[token_positions, selected_slots].unsqueeze(-1)

            # 把专家输出乘以权重
            weighted_output = expert_output * weights

            # 把结果加回对应 token 的位置
            #
            # 为什么用 index_add_？
            #
            # 因为一个 token 可能选择多个专家。
            # 例如 token 0 选择了专家 2 和专家 0。
            # 那么 token 0 的最终输出是:
            #
            #   output[0] =
            #       weight_2 * expert_2(x[0])
            #     + weight_0 * expert_0(x[0])
            #
            # index_add_ 就是在做这个“加回原位置”的操作。
            output_flat.index_add_(
                0,
                token_positions,
                weighted_output,
            )

        # --------------------------------------------------
        # 8. 恢复原始形状
        # --------------------------------------------------
        # output:
        #   [batch_size, seq_len, hidden_dim]
        output = output_flat.reshape(batch_size, seq_len, hidden_dim)

        if return_router_info:
            return output, {
                "router_logits": router_logits.reshape(
                    batch_size,
                    seq_len,
                    self.num_experts,
                ),
                "router_probs": router_probs.reshape(
                    batch_size,
                    seq_len,
                    self.num_experts,
                ),
                "topk_indices": topk_indices.reshape(
                    batch_size,
                    seq_len,
                    self.top_k,
                ),
                "topk_weights": topk_weights.reshape(
                    batch_size,
                    seq_len,
                    self.top_k,
                ),
            }

        return output


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 3
    hidden_dim = 8
    num_experts = 4
    top_k = 2

    moe = SimpleMoE(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        ffn_dim=32,
    )

    # 假设这是 Transformer 某一层里面的 hidden states
    #
    # x.shape:
    #   [batch_size, seq_len, hidden_dim]
    x = torch.randn(batch_size, seq_len, hidden_dim)

    y, info = moe(x, return_router_info=True)

    print("输入 x shape:")
    print(x.shape)

    print("\n输出 y shape:")
    print(y.shape)

    print("\n每个 token 选择的专家 top-k:")
    print(info["topk_indices"])

    print("\n每个 token 对应 top-k 专家的权重:")
    print(info["topk_weights"])

    print("\n第 0 个 batch、第 0 个 token 对所有专家的 router 概率:")
    print(info["router_probs"][0, 0])

    counts = torch.bincount(
        info["topk_indices"].reshape(-1),
        minlength=num_experts,
    )

    print("\n每个专家被选中的次数:")
    for expert_id, count in enumerate(counts.tolist()):
        print(f"Expert {expert_id}: {count} 次")