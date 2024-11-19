import torch
from einops import rearrange

torch.manual_seed(123)
import numpy as np
np.random.seed(123)
# R input N*d
# (R-mean)/norm2
def normalize(R):

    mean = R.mean(dim=0)
    R = R - mean
    norms = torch.norm(R, p=2, dim=1, keepdim=True)
    R = R / norms
    return R


# N,d,
# 协方差矩阵的求和放在了里面，也就是利用外机，d1=
def cal_cov(R):
    Z = torch.nn.functional.normalize(R, dim=1)
    A = torch.matmul(Z.T, Z) / Z.shape[0]
    return A


def cal_entropy(A):

    # 返回U,S,V,除以 trace(A)也就是归一化因子，也就是不用再求和了
    eig_val = torch.svd(A / torch.trace(A))[1]
    # 去除了  .item()
    entropy = -(eig_val * torch.log(eig_val)).nansum()
    return entropy

def cal_entropy_che(A,k=10,degree=3):
    A=cal_cov(normalize(A))
    max_sing=chebyshev_polynomial(A,k,degree)
    entropy=-torch.sigmoid((max_sing*torch.log(max_sing)))
    return entropy    
    

def compute(R):
    return cal_entropy(cal_cov(normalize(R)))


def chebyshev_polynomial(A, k, degree):
    """
    使用切比雪夫多项式近似矩阵的最大奇异值
    :param A: 输入矩阵
    :param k: 迭代次数
    :param degree: 多项式的最高次数
    :return: 近似的最大奇异值
    """
    n = A.size(0)
    v = torch.randn(n, 1, device=A.device)
    v = v / torch.norm(v)

    for _ in range(k):
        w = A @ v
        w = A.t() @ w
        v = w / torch.norm(w)

    lambda_max = torch.norm(A @ v) / torch.norm(v)
    
    # 切比雪夫多项式
    T = [v, A @ v / lambda_max]
    for i in range(2, degree + 1):
        T.append(2 * (A @ T[-1] / lambda_max) - T[-2])

    # 估计最大奇异值
    approx_singular_value = lambda_max * torch.max(torch.abs(T[-1]))
    return approx_singular_value

# 示例矩阵
A = torch.randn(100, 100, device='cuda')
k = 100  # 迭代次数
degree = 10  # 多项式的最高次数

approx_singular_value = chebyshev_polynomial(A, k, degree)
print("Approximate maximum singular value:", approx_singular_value.item())



from tqdm import tqdm


def train_eRank(
    better_model, worse_model, train_loader, optimizer, train_args, writer, **kwargs
):
    for epoch in tqdm(range(train_args.eRank_epochs), desc="train eRank epochs"):
        for x, y, depth in tqdm(
            train_loader, total=len(train_loader), desc="train eRank"
        ):
            x = x.unsqueeze(dim=-1).to(train_args.device)
            _, features = better_model(x, return_features=True)
            # print(features.shape)
            _, worse_features = worse_model(x, return_features=True)
            features = rearrange(features, "b l d ->(b l) d")
            worse_features = rearrange(worse_features, "b l d ->(b l) d")
            R2 = cal_entropy_che(worse_features)
            R1 = cal_entropy_che(features)
            loss = (R1 - R2) / torch.sqrt(R1**2 + R2**2)
            writer.add_scalar(
                "eRank_loss_e_layer_{e_layer}_lr_{lr}_d_model_{d_model}".format(
                    **kwargs
                ),
                loss.item(),
                epoch,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return better_model, worse_model
