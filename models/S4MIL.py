# This code is taken from the original S4 repository https://github.com/HazyResearch/state-spaces
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import opt_einsum as oe

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X


class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = _r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L)  # (H L)
        u_f = torch.fft.rfft(u.to(torch.float32), n=2*L)  # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y


class S4Model(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False, L=512, in_dim_titan=768, in_dim_uni=1536):
        super(S4Model, self).__init__()
        self.n_classes = n_classes
        self._fc1 = [nn.Linear(L, L)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
            print("dropout: ", dropout)
        self._fc1 = nn.Sequential(*self._fc1)
        self.s4_block = nn.Sequential(nn.LayerNorm(512),
                                      S4D(d_model=512, d_state=32, transposed=False))

        self.classifier = nn.Linear(512, self.n_classes)
        self.survival = survival
        # Titan特征处理
        self.titan_proj = nn.Sequential(
            nn.Linear(in_dim_titan, L),
            nn.LayerNorm(L),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Uni特征处理
        self.uni_proj = nn.Sequential(
            nn.Linear(in_dim_uni, L),
            nn.LayerNorm(L),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x1, x2):
        # 处理Titan特征，使用memory efficient的方式
        x1 = x1.unsqueeze(1)  # [B, 1, 768]
        x1 = self.titan_proj(x1)  # [B, 1, L]
        
        # 分批处理Uni特征以减少内存使用
        B = x1.size(0)
        chunk_size = 512  # 可以根据实际GPU内存调整这个值
        num_chunks = (x2.size(0) + chunk_size - 1) // chunk_size
        x2_processed = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, x2.size(0))
            chunk = x2[start_idx:end_idx]
            chunk = chunk.unsqueeze(0).expand(B, -1, -1)
            chunk = self.uni_proj(chunk)
            x2_processed.append(chunk)
            
            # 主动清理不需要的中间变量
            torch.cuda.empty_cache()
        
        x2 = torch.cat(x2_processed, dim=1)
        
        # 特征融合
        x = torch.cat([x1, x2], dim=1)
        
        # 使用gradient checkpointing来节省内存
        if self.training:
            x = torch.utils.checkpoint.checkpoint(self._fc1, x)
            x = torch.utils.checkpoint.checkpoint(self.s4_block, x)
        else:
            x = self._fc1(x)
            x = self.s4_block(x)
            
        x = torch.max(x, axis=1).values
        logits = self.classifier(x)
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        return logits, Y_prob, Y_hat, A_raw, results_dict


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.s4_block  = self.s4_block .to(device)
        self.classifier = self.classifier.to(device)
        
if __name__ == "__main__":
    data = torch.randn((6000, 1024))
    data.to('cuda')
    # model1 = TransMIL_l_v2(input_dim = 1024, layer =4, n_classes = 4, act = 'gelu', dropout = True)
    model = S4Model(in_dim = 1024, n_classes = 4, act = 'gelu', dropout = 0.25)
    print(model)
    results_dict = model(data)
    print(results_dict)