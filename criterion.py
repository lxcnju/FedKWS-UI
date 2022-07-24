import torch


def guassian_kernel(
        xs, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
    n_samples = int(xs.size()[0])

    L2_distance = ((
        xs.unsqueeze(dim=1) - xs.unsqueeze(dim=0)
    ) ** 2).sum(dim=2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth += 1e-8

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / band) for band in bandwidth_list
    ]
    return sum(kernel_val)


def hsic(Kx, Ky):
    Kxy = torch.mm(Kx, Ky.transpose(0, 1))
    n = Kxy.shape[0]
    v1 = torch.sum(torch.diag(Kxy)) / n ** 2
    v2 = torch.mean(Kx) * torch.mean(Ky)
    v3 = 2.0 * torch.mean(Kxy) / n
    v = (v1 + v2 - v3) * n ** 2 / (n - 1) ** 2
    return v


def calculate_hsic_loss(xs1, xs2):
    kx1 = guassian_kernel(xs1)
    kx2 = guassian_kernel(xs2)
    bs = xs1.shape[0]
    h_mat = torch.eye(bs) - 1.0 / bs * torch.ones(bs, bs)
    h_mat = h_mat.to(xs1.device)

    loss = torch.diag(
        kx1.mm(h_mat).mm(kx2).mm(h_mat)
    ).sum() / ((bs - 1) ** 2)

    return loss


def soft_cross_entropy(logits, labels, mu=0.0):
    bs, nc = logits.shape

    onehot_mat = torch.diag(torch.ones(nc)).to(labels.device)
    onehot_labels = onehot_mat[labels]
    onehot_labels = onehot_labels * (1.0 - mu) + mu / nc

    losses = -1.0 * onehot_labels * logits.log_softmax(dim=-1)
    loss = losses.sum(dim=1).mean()
    return loss


if __name__ == "__main__":
    bs = 2
    nc = 10
    logits = torch.randn(bs, nc)
    labels = torch.LongTensor([3, 7])

    loss = soft_cross_entropy(logits, labels, mu=0.2)
    loss = soft_cross_entropy(logits, labels, mu=0.0)
    loss = soft_cross_entropy(logits, labels, mu=-0.2)

    xs1 = torch.randn(bs, nc)
    xs2 = torch.randn(bs, nc)
    hsic_loss = calculate_hsic_loss(xs1, xs2)
    print(hsic_loss)
