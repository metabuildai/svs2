import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import utils


class VarianceAdaptor(nn.Module):
    def __init__(self, encoder_hidden, variance_predictor_filter_size, variance_predictor_kernel_size, variance_predictor_dropout, log_offset):
        super(VarianceAdaptor, self).__init__()
        self.log_offset = log_offset
        self.length_regulator = LengthRegulator()

    def forward(self, x, src_mask, mel_mask, duration_target, pitch_target, max_len):
        device = x.device

        x, mel_len = self.length_regulator(x, duration_target, max_len)
        mel_len = mel_len.to(device)
        
        return x, mel_len, mel_mask

    def inference(self, x, src_mask, duration, pitch, max_len=None):
        device = x.device

        x, mel_len = self.length_regulator(x, duration, max_len)
        mel_len = mel_len.to(device)
        mel_mask = utils.get_mask_from_lengths(mel_len)

        return x, mel_mask


class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = utils.pad(output, max_len)
        else:
            output = utils.pad(output)

        return output, torch.LongTensor(mel_len)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    def __init__(self, encoder_hidden, variance_predictor_filter_size, variance_predictor_kernel_size, variance_predictor_dropout):
        super(VariancePredictor, self).__init__()
        self.conv_layer = nn.Sequential(
            Conv(encoder_hidden, variance_predictor_filter_size, kernel_size=variance_predictor_kernel_size, padding=(variance_predictor_kernel_size-1)//2),
            nn.ReLU(),
            nn.LayerNorm(variance_predictor_filter_size),
            nn.Dropout(variance_predictor_dropout),
            Conv(variance_predictor_filter_size, variance_predictor_filter_size, kernel_size=variance_predictor_kernel_size, padding=1),
            nn.ReLU(),
            nn.LayerNorm(variance_predictor_filter_size),
            nn.Dropout(variance_predictor_dropout))
        self.linear_layer = nn.Linear(variance_predictor_filter_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.)

        return out


class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    def __init__(self, n_src_vocab, len_max_seq, d_word_vec, n_head, d_k, d_v, d_model, d_inner, fft_kernel_size, dropout):
        super(Encoder, self).__init__()
        self.len_max_seq = len_max_seq
        self.d_word_vec = d_word_vec
        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, 0)
        self.pitch_emb = nn.Embedding(1600, d_word_vec, 0)

    def forward(self, src_seq, mask, d_seq, p_seq):
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        output = torch.cat((self.src_word_emb(src_seq), self.pitch_emb(p_seq)), dim=2)

        return output


class Decoder(nn.Module):
    def __init__(self,
                 len_max_seq,
                 d_word_vec,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 fft_kernel_size,
                 dropout):
        super(Decoder, self).__init__()
        self.max_seq_len = len_max_seq
        self.decoder_hidden = n_head
        n_position = len_max_seq + 1

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0), requires_grad=False)

        self.fft0 = FFTBlock(d_model, d_inner, n_head, d_k, d_v, fft_kernel_size, dropout=dropout)
        self.fft1 = FFTBlock(d_model, d_inner, n_head, d_k, d_v, fft_kernel_size, dropout=dropout)
        self.fft2 = FFTBlock(d_model, d_inner, n_head, d_k, d_v, fft_kernel_size, dropout=dropout)
        self.fft3 = FFTBlock(d_model, d_inner, n_head, d_k, d_v, fft_kernel_size, dropout=dropout)

    def forward(self, enc_seq, mask):
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        output = enc_seq + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        output, _ = self.fft0(output, mask, slf_attn_mask)
        output, _ = self.fft1(output, mask, slf_attn_mask)
        output, _ = self.fft2(output, mask, slf_attn_mask)
        output, _ = self.fft3(output, mask, slf_attn_mask)

        return output


class FFTBlock(torch.nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 kernel_size,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, kernel_size, dropout=dropout)

    def forward(self, enc_input, mask, slf_attn_mask):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class ConvNorm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class PostNet(nn.Module):
    def __init__(self,
                 n_mel_channels=80,
                 postnet_embedding_dim=512,
                 postnet_kernel_size=5):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.conv0 = nn.Sequential(
            ConvNorm(n_mel_channels,
                     postnet_embedding_dim,
                     kernel_size=postnet_kernel_size,
                     stride=1,
                     padding=int((postnet_kernel_size - 1) / 2),
                     dilation=1),
            nn.BatchNorm1d(postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout()
        )
        self.conv1 = nn.Sequential(
            ConvNorm(postnet_embedding_dim,
                     postnet_embedding_dim,
                     kernel_size=postnet_kernel_size,
                     stride=1,
                     padding=int((postnet_kernel_size - 1) / 2),
                     dilation=1),
            nn.BatchNorm1d(postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout()
        )
        self.conv2 = nn.Sequential(
            ConvNorm(postnet_embedding_dim,
                     postnet_embedding_dim,
                     kernel_size=postnet_kernel_size,
                     stride=1,
                     padding=int((postnet_kernel_size - 1) / 2),
                     dilation=1),
            nn.BatchNorm1d(postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout()
        )
        self.conv3 = nn.Sequential(
            ConvNorm(postnet_embedding_dim,
                     postnet_embedding_dim,
                     kernel_size=postnet_kernel_size,
                     stride=1,
                     padding=int((postnet_kernel_size - 1) / 2),
                     dilation=1),
            nn.BatchNorm1d(postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout()
        )
        self.conv4 = nn.Sequential(
            ConvNorm(postnet_embedding_dim,
                     n_mel_channels,
                     kernel_size=postnet_kernel_size,
                     stride=1,
                     padding=int((postnet_kernel_size - 1) / 2),
                     dilation=1),
            nn.BatchNorm1d(n_mel_channels),
            nn.Dropout()
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.contiguous().transpose(1, 2)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v)  # (n*b) x lv x dv

        mask = mask.repeat(self.n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(self.n_head, sz_b, len_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, kernel, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=kernel[0], padding=(kernel[0] - 1) // 2)
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=kernel[1], padding=(kernel[1] - 1) // 2)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Conv1d(num_features, num_hidden, 5, padding=2)
        self.fc2 = nn.Conv1d(num_hidden, num_features, 5, padding=2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, d_model, seq_len, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = FeedForward(seq_len, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, d_model, seq_len)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, seq_len, d_model)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, seq_len, d_model)
        out = x + residual
        return out


class MixerBlock(nn.Module):
    def __init__(self, d_model=256, seq_len=256, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.token_mixer = TokenMixer(d_model, seq_len, expansion_factor, dropout)
        self.channel_mixer = ChannelMixer(d_model, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        x = self.token_mixer(x)
        # x = self.channel_mixer(x)
        # x.shape == (batch_size, seq_len, d_model)
        return x