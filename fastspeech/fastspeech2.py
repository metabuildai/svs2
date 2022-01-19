import torch.nn as nn
from module import VarianceAdaptor, Encoder, Decoder, PostNet, MixerBlock


class FastSpeech2(nn.Module):
    def __init__(self, decoder_hidden, n_mel_channels, src_vocab, max_seq_len, encoder_hidden, encoder_head,
                 decoder_head, decoder_dropout, fft_filter_size,  fft_kernel_size, e_dropout, log_offset,
                 variance_predictor_filter_size, variance_predictor_kernel_size, variance_predictor_dropout):
        super(FastSpeech2, self).__init__()
        self.encoder = Encoder(src_vocab, max_seq_len, encoder_hidden, encoder_head, encoder_hidden // encoder_head,
                               encoder_hidden // encoder_head, encoder_hidden * 2, fft_filter_size, fft_kernel_size, e_dropout)
        self.mel_linear = nn.Linear(decoder_hidden * 2, n_mel_channels)
        self.mixter = nn.Sequential(
            MixerBlock(decoder_hidden * 2, decoder_hidden * 2),
            MixerBlock(decoder_hidden * 2, decoder_hidden * 2)
        )

    def forward(self, src_seq, d_target, p_target, std_p_target, max_mel_len, src_mask, mel_mask):

        max_mel_len = max_mel_len.item()

        encoder_output = self.encoder(src_seq, src_mask, d_target, p_target)
        encoder_output = self.mixter(encoder_output)

        mel_output = self.mel_linear(encoder_output)

        return mel_output, mel_output, src_mask, mel_mask

    def inference(self, src_seq, duration, pitch, std_pitch, src_mask):
        encoder_output = self.encoder(src_seq, src_mask, duration, pitch)
        encoder_output = self.mixter(encoder_output)

        mel_output = self.mel_linear(encoder_output)

        return mel_output

