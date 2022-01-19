import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, mel, mel_postnet, mel_target, src_mask, mel_mask):
        mel_target.requires_grad = False

        mel = mel.masked_select(mel_mask.unsqueeze(-1))
        mel_postnet = mel_postnet.masked_select(mel_mask.unsqueeze(-1))
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

        return mel_loss, mel_postnet_loss
