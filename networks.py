import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import weights_init


def load_model(args):
    if args.net == "ResNet":
        model = AudioResNet(
            n_layer=15,
            n_classes=args.n_classes,
            n_channel=45,
        )
    elif args.net == "DSCNN":
        model = AudioDSCNN(
            n_channel=172,
            n_classes=args.n_classes,
        )
    elif args.net == "MHAttRNN":
        model = AudioMHAttRNN(
            input_dim=args.input_channel,
            n_classes=args.n_classes,
        )
    elif args.net == "Transformer":
        model = AudioTransformer(
            input_size=(args.n_time, args.input_channel),
            patch_size=(8, 8),
            stride=(4, 4),
            n_classes=args.n_classes,
            n_layer=4,
            d_model=80,
        )
    else:
        raise ValueError("No such net.")

    model.apply(weights_init)
    return model


class DepthSeparableConv(nn.Module):
    def __init__(
            self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_planes,
        )
        self.conv2 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1
        )

    def forward(self, xs):
        out = self.conv1(xs)
        out = self.conv2(out)
        return out


class AudioDSCNN(nn.Module):
    def __init__(self, n_channel=172, n_classes=35):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(1, n_channel, kernel_size=(10, 4), stride=(2, 2)),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            DepthSeparableConv(
                n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            DepthSeparableConv(
                n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            DepthSeparableConv(
                n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            DepthSeparableConv(
                n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            DepthSeparableConv(
                n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_channel, n_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)

        out = self.encoder(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class AudioMHAttRNN(nn.Module):
    def __init__(self, input_dim=40, n_classes=35, n_head=4, n_hidden=80):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_head = n_head
        self.n_hidden = n_hidden

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 10, (5, 1), stride=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 1, (5, 1), stride=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=n_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.q_proj_layers = nn.ModuleList()
        self.k_proj_layers = nn.ModuleList()

        d = n_hidden * 2
        d0 = int(d / n_head)
        for _ in range(n_head):
            self.q_proj_layers.append(
                nn.Linear(d, d0, bias=False)
            )
            self.k_proj_layers.append(
                nn.Linear(d, d0, bias=False)
            )

        self.classifier = nn.Linear(d, n_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)

        out = self.cnn(x)

        # (bs, 1, time, freq)
        out = out.permute(0, 2, 1, 3)

        bs, nt = out.shape[0], out.shape[1]
        out = out.reshape((bs, nt, -1))

        # (bs, time, d)
        h0 = torch.zeros(4, bs, self.n_hidden).to(x.device)
        out, _ = self.rnn(out, h0)

        mid = int(nt / 2)

        query = out[:, mid, :]  # (bs, d)

        # multi head attention
        outputs = []
        for k in range(self.n_head):
            proj_out = self.k_proj_layers[k](out)
            proj_query = self.q_proj_layers[k](query)

            weights = torch.bmm(
                proj_out, proj_query.unsqueeze(dim=2)
            ).softmax(dim=1)

            attn_out = torch.bmm(
                weights.permute(0, 2, 1), proj_out
            ).squeeze(dim=1)
            outputs.append(attn_out)
        out = torch.cat(outputs, dim=1)

        logits = self.classifier(out)
        return out, logits


class AudioBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, paddings, dilations):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=1,
            padding=paddings[0],
            dilation=dilations[0],
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, affine=False)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1,
            padding=paddings[1],
            dilation=dilations[1],
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, affine=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + x
        out = F.relu(out)
        return out


class AudioResNet(nn.Module):
    def __init__(self, n_layer=15, n_classes=35, n_channel=45):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes

        self.conv0 = nn.Conv2d(
            1, n_channel, kernel_size=3, padding=1, bias=False
        )
        self.bn0 = nn.BatchNorm2d(n_channel, affine=False)

        assert ((n_layer - 3) % 2 == 0), "AudioResNet depth is 2n+3"
        n = int((n_layer - 3) / 2)

        self.in_planes = n_channel

        self.layers = nn.ModuleList()

        for b in range(n):
            p0 = int(2 ** (2 * b // 3))
            p1 = int(2 ** ((2 * b + 1) // 3))
            layer = AudioBasicBlock(
                in_planes=n_channel,
                planes=n_channel,
                paddings=[p0, p1],
                dilations=[p0, p1],
            )
            self.layers.append(layer)

        self.last_conv = nn.Conv2d(
            n_channel, n_channel,
            kernel_size=3,
            padding=int(2 ** (2 * n // 3)),
            dilation=int(2 ** (2 * n // 3)),
            bias=False
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            45, n_classes
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)

        out = F.relu(self.bn0(self.conv0(x)))
        for layer in self.layers:
            out = layer(out)

        out = self.last_conv(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return out, logits


class PatchEmbed(nn.Module):
    def __init__(
            self, input_size, patch_size,
            stride, in_chans=1, embed_dim=64):
        super().__init__()

        w, h = input_size
        kw, kh = patch_size
        sw, sh = stride
        pw, ph = 0, 0
        ow = int((w + 2 * pw - kw) / sw + 1)
        oh = int((h + 2 * ph - kh) / sh + 1)
        self.n_patch = ow * oh

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class AudioTransformer(nn.Module):
    def __init__(
            self, input_size, n_layer=4, n_classes=35,
            patch_size=(8, 8), stride=(4, 4), d_model=96):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes

        self.patch_embed = PatchEmbed(
            input_size=input_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=1,
            embed_dim=d_model,
        )

        self.n_patch = self.patch_embed.n_patch

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patch + 1, d_model),
            requires_grad=True
        )

        self.cls_embedding = nn.Parameter(
            torch.randn(1, 1, d_model),
            requires_grad=True
        )

        # batch_first=False
        basic_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            nhead=4,
        )
        self.encoder = nn.TransformerEncoder(
            basic_layer,
            num_layers=n_layer
        )

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)

        out = self.patch_embed(x)
        bs = out.shape[0]

        out = torch.cat([
            self.cls_embedding.repeat((bs, 1, 1)),
            out
        ], dim=1)
        out += self.pos_embedding

        # (bs, time, d) --> (time, bs, d)
        out = out.permute(1, 0, 2)
        out = self.encoder(out)
        out = out.permute(1, 0, 2)

        out = out[:, 0, :]
        logits = self.classifier(out)
        return out, logits


if __name__ == "__main__":
    for n_classes in [12, 35]:
        for net in ["DSCNN", "MHAttRNN", "ResNet", "Transformer"]:
            if net == "ResNet":
                model = AudioResNet(
                    n_layer=15,
                    n_classes=n_classes,
                    n_channel=45,
                )
            elif net == "DSCNN":
                model = AudioDSCNN(
                    n_channel=172,
                    n_classes=n_classes,
                )
            elif net == "MHAttRNN":
                model = AudioMHAttRNN(
                    input_dim=40,
                    n_classes=n_classes,
                )
            elif net == "Transformer":
                model = AudioTransformer(
                    input_size=(101, 40),
                    patch_size=(8, 8),
                    stride=(4, 4),
                    n_classes=n_classes,
                    n_layer=4,
                    d_model=80,
                )
            else:
                raise ValueError("No such net.")

            n_params = sum([
                param.numel() for param in model.parameters()
            ])
            print("Total number of parameters : {}".format(n_params))
