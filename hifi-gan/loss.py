import torch
from torch.nn.functional import l1_loss, mse_loss


def discriminator_loss(real_mpd, fake_mpd, real_msd, fake_msd):
    period_loss = 0
    for real, fake in zip(real_mpd, fake_mpd):
        period_loss += mse_loss(real, torch.ones_like(real)) + mse_loss(fake, torch.zeros_like(fake))

    scale_loss = 0
    for real, fake in zip(real_msd, fake_msd):
        scale_loss += mse_loss(real, torch.ones_like(real)) + mse_loss(fake, torch.zeros_like(fake))

    return period_loss / (len(real_mpd) * 2), scale_loss / (len(real_msd) * 2)


def generator_loss(real_mel, fake_mel):
    return l1_loss(fake_mel, real_mel)


def adversarial_loss(fake_mpd, real_mpd, fake_msd, real_msd, real_mpd_feature, fake_mpd_feature, real_msd_feature, fake_msd_feature):
    mpd_loss = msd_loss = 0

    for fake, real in zip(fake_mpd, real_mpd):
        mpd_loss += l1_loss(fake, real)

    for real, fake in zip(real_mpd_feature, fake_mpd_feature):
        mpd_loss += l1_loss(fake, real)

    for fake, real in zip(fake_msd, real_msd):
        msd_loss += l1_loss(fake, real)

    for real, fake in zip(real_msd_feature, fake_msd_feature):
        msd_loss += l1_loss(fake, real)

    return mpd_loss / (len(fake_mpd) + len(fake_mpd_feature)), msd_loss / (len(fake_msd) + len(fake_msd_feature))
