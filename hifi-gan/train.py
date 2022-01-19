import itertools
import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from preference import Config, load, save
import tqdm
import pickle
from data import Dataset
from utils import MelSpectrogram
import numpy as np
from loss import discriminator_loss, generator_loss, adversarial_loss
import logging
import sys


def setup(config: Config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(config.gpu_index)

    if torch.cuda.is_available() and config.gpu_index >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print("checkpoint directory : ", config.checkpoint_dir)

    path = os.path.join(config.log_dir, 'train')
    os.makedirs(path, exist_ok=True)
    print('train log directory: ', path)
    config.train_log = path

    path = os.path.join(config.log_dir, 'valid')
    os.makedirs(path, exist_ok=True)
    print('valid log directory : ', path)
    config.valid_log = path

    if config.use_log is True:
        sw_train = SummaryWriter(config.train_log)
        sw_valid = SummaryWriter(config.valid_log)
    else:
        sw_train = None
        sw_valid = None

    print('{s:{c}^{n}}\n'.format(s='complete: setup step', n=50, c='-'))

    return device, sw_train, sw_valid


def build_dataloader(files, cf: Config, desc):
    audio_list = []
    for file in tqdm.tqdm(files, desc=desc):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            audio_list.append(np.array(data['audio']))
    dataset = Dataset(audio_list, cf.audio_segment_size, cf.audio_segment_size // 2)
    dataloader = DataLoader(dataset,
                            shuffle=cf.shuffle,
                            batch_size=cf.batch_size,
                            num_workers=cf.num_workers,
                            pin_memory=cf.pin_memory,
                            drop_last=cf.drop_last)

    return dataloader


def build_dataset(cf: Config):
    train_loader = build_dataloader(cf.train_file, cf, 'load train dataset')

    if cf.use_valid and len(cf.valid_file) > 0:
        valid_loader = build_dataloader(cf.valid_file, cf, 'load valid dataset')
    else:
        valid_loader = None

    print('shuffle: ', cf.shuffle)
    print('batch size: ', cf.batch_size)
    print('workers: ', cf.num_workers)
    print('pin memory: ', cf.pin_memory)
    print('drop last: ', cf.drop_last)

    print('{s:{c}^{n}}\n'.format(s='complete: dataset step', n=50, c='-'))

    return train_loader, valid_loader


def build_model(config: Config, device: torch.device):
    generator = build_generator(config)
    generator = generator.to(device)
    generator = torch.jit.script(generator)
    print('--Generator')

    mpd = MultiPeriodDiscriminator().to(device)
    mpd = torch.jit.script(mpd)
    print('--MPD')

    msd = MultiScaleDiscriminator().to(device)
    msd = torch.jit.script(msd)
    print('--MSD')

    optim_g = torch.optim.AdamW(generator.parameters(), config.learning_rate, betas=(config.adam_b1, config.adam_b2))
    print('--Generator AdamW Optimizer')
    print('learning rate: ', config.learning_rate)
    print('betas: ', [config.adam_b1, config.adam_b2])

    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), config.learning_rate, betas=(config.adam_b1, config.adam_b2))
    print('--Discriminator AdamW Optimizer')
    print('learning rate: ', config.learning_rate)
    print('betas: ', [config.adam_b1, config.adam_b2])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.lr_decay, last_epoch=-1)
    print('--Generator Optimizer Exponential Scheduler')
    print('gamma: ', config.lr_decay)

    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.lr_decay, last_epoch=-1)
    print('--Discriminator Optimizer Exponential Scheduler')
    print('gamma: ', config.lr_decay)

    print('{s:{c}^{n}}\n'.format(s='complete: model step', n=50, c='-'))

    return generator, mpd, msd, optim_g, optim_d, scheduler_g, scheduler_d


def build_generator(config: Config):
    generator = Generator(
        resblock_kernel_size=config.resblock_kernel_size,
        upsample_rate=config.upsample_rate,
        upsample_kernel_size=config.upsample_kernel_size,
        upsample_initial_channel=config.upsample_initial_channel,
        resblock_dilation_size=config.resblock_dilation_size,
        num_mel=config.num_mel
    )

    return generator


def validation(dataloader, generator: Generator, mpd: MultiPeriodDiscriminator, msd: MultiScaleDiscriminator, mel_spectogram, cf: Config):
    sum_period_dis_loss = 0
    sum_scale_dis_loss = 0
    sum_gen_loss = 0
    sum_mpd_adv_loss = 0
    sum_msd_adv_loss = 0
    total = 0

    generator.eval()
    mpd.eval()
    msd.eval()
    with torch.no_grad():
        for real_wave in dataloader:
            real_wave = real_wave.to(device, non_blocking=True)

            real_mel = mel_spectogram(real_wave.squeeze(1))

            fake_wave = generator(real_mel)
            fake_mel = mel_spectogram(fake_wave.squeeze(1))

            real_wave = real_wave.unsqueeze(1)
            real_mpd, real_mpd_feature = mpd(real_wave)
            fake_mpd, fake_mpd_feature = mpd(fake_wave)

            real_msd, real_msd_feature = msd(real_wave)
            fake_msd, fake_msd_feature = msd(fake_wave)

            period_dis_loss, scale_dis_loss = discriminator_loss(real_mpd, fake_mpd, real_msd, fake_msd)
            loss_gen = generator_loss(real_mel, fake_mel)
            mpd_adv_loss, msd_adv_loss = adversarial_loss(fake_mpd, real_mpd, fake_msd, real_msd, real_mpd_feature,
                                                          fake_mpd_feature, real_msd_feature, fake_msd_feature)

            size = real_wave.shape[0]
            sum_period_dis_loss += period_dis_loss.item() * size
            sum_scale_dis_loss += scale_dis_loss.item() * size
            sum_gen_loss += loss_gen.item() * size
            sum_mpd_adv_loss += mpd_adv_loss.item() * size
            sum_msd_adv_loss += msd_adv_loss.item() * size
            total += size
    generator.train()
    mpd.train()
    msd.train()

    return sum_gen_loss / total, sum_period_dis_loss / total, sum_scale_dis_loss / total, sum_mpd_adv_loss / total, sum_msd_adv_loss / total


def summary(sw: SummaryWriter, gen, dis_period, dis_scale, adv_period, adv_scale, step):
    sw.add_scalar('hifi-gan/gen', gen, step)
    sw.add_scalar('hifi-gan/dis_period', dis_period, step)
    sw.add_scalar('hifi-gan/dis_scale', dis_scale, step)
    sw.add_scalar('hifi-gan/adv_period', adv_period, step)
    sw.add_scalar('hifi-gan/adv_scale', adv_scale, step)


def train_model(
        dataloader,
        valid_loader,
        generator,
        mpd,
        msd,
        optim_g,
        optim_d,
        scheduler_g,
        scheduler_d,
        device,
        sw_train,
        sw_valid,
        cf: Config,
        logger,
        start_epoch=0
):
    mel_spectogram = MelSpectrogram(cf.sampling_rate, cf.n_fft, cf.num_mel, cf.win_size, cf.hop_size)
    mel_spectogram = mel_spectogram.to(device)

    generator.train()
    mpd.train()
    msd.train()
    print('max epoch', cf.training_epoch)
    for epoch in range(start_epoch, cf.training_epoch):
        sum_gen = 0
        sum_dis_period = 0
        sum_dis_scale = 0
        sum_adv_period = 0
        sum_adv_scale = 0
        total = 0
        for real_wave in dataloader:
            real_wave = real_wave.to(device)
            real_mel = mel_spectogram(real_wave.squeeze(1))

            fake_wave = generator(real_mel)
            fake_mel = mel_spectogram(fake_wave.squeeze(1))

            # train discriminator
            real_wave = real_wave.unsqueeze(1)
            real_mpd, _ = mpd(real_wave)
            fake_mpd, _ = mpd(fake_wave.detach())

            real_msd, _ = msd(real_wave)
            fake_msd, _ = msd(fake_wave.detach())

            loss_dis_period, loss_dis_scale = discriminator_loss(real_mpd, fake_mpd, real_msd, fake_msd)
            loss_dis_period.backward(retain_graph=True)
            loss_dis_scale.backward()
            optim_d.step()

            optim_d.zero_grad()

            # train generator
            real_mpd, real_mpd_feature = mpd(real_wave)
            fake_mpd, fake_mpd_feature = mpd(fake_wave)

            real_msd, real_msd_feature = msd(real_wave)
            fake_msd, fake_msd_feature = msd(fake_wave)

            loss_gen = generator_loss(real_mel, fake_mel)
            loss_adv_period, loss_adv_scale = adversarial_loss(fake_mpd, real_mpd, fake_msd, real_msd, real_mpd_feature, fake_mpd_feature, real_msd_feature, fake_msd_feature)
            loss_gen.backward(retain_graph=True)
            loss_adv_period.backward(retain_graph=True)
            loss_adv_scale.backward()
            optim_g.step()

            optim_g.zero_grad()

            size = real_wave.shape[0]
            sum_dis_period += loss_dis_period.item() * size
            sum_dis_scale += loss_dis_scale.item() * size
            sum_gen += loss_gen.item() * size
            sum_adv_period += loss_adv_period.item() * size
            sum_adv_scale += loss_adv_scale.item() * size
            total += size
        # end of train epoch
        train_dis_period = sum_dis_period / total
        train_dis_scale = sum_dis_scale / total
        train_gen = sum_gen / total
        train_adv_period = sum_adv_period / total
        train_adv_scale = sum_adv_scale / total

        if valid_loader is not None:
            valid_gen, valid_dis_period, valid_dis_scale, valid_adv_period, valid_adv_scale = validation(
                valid_loader, generator, mpd, msd, mel_spectogram, cf
            )
            print('epoch: {}, train gen loss: {:4.3f}, valid gen loss: {:4.3f}'.format(
                epoch + 1, train_gen, valid_gen))
            logger.info('epoch: {}, train gen loss: {:4.3f}, valid gen loss: {:4.3f}'.format(epoch + 1, train_gen, valid_gen))
        else:
            print('epoch: {}, gen loss: {:4.3f}'.format(epoch + 1, train_gen))
        # end of epoch
        cf.last_epoch = epoch

        # logging
        if config.use_log:
            summary(sw_train, train_gen, train_dis_period, train_dis_scale, train_adv_period, train_adv_scale, epoch + 1)
            if valid_loader is not None:
                summary(sw_valid, valid_gen, valid_dis_period, valid_dis_scale, valid_adv_period, valid_adv_scale,
                        epoch + 1)

        # checkpoint
        if (epoch + 1) % cf.checkpoint_interval == 0:
            path = os.path.join(cf.checkpoint_dir, '{}.tar'.format(epoch + 1))
            torch.save(
                {'generator': generator.state_dict(),
                 'mpd': mpd.state_dict(),
                 'msd': msd.state_dict(),
                 'optim_g': optim_g.state_dict(),
                 'optim_d': optim_d.state_dict(),
                 'epoch': epoch},
                path
            )
            cf.last_checkpoint_file = path
            print('save checkpoint : ', path)

            path = os.path.join(cf.checkpoint_dir, 'config.json')
            save(cf, path)
            print('save config : ', path)
        scheduler_g.step()
        scheduler_d.step()


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S %Z%z')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    cmd = ' '.join(sys.argv)
    logger.info('python ' + cmd)

    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='JSON file for configuration')
    parser.add_argument('--epoch', '-e', type=int, help='num of train loop')
    parser.add_argument('--batch', '-b', type=int, help='batch size for training')
    parser.add_argument('--rate', type=float, help='learning rate for training')
    parser.add_argument('--checkpoint', '-c', help='directory for save checkpoint file')
    parser.add_argument('--interval', '-t', type=int, help='check point save interval time(sec)')
    parser.add_argument('--model', '-m', help='model path for fine tuning')
    parser.add_argument('--gpu', '-g', type=int, help='gpu device index')
    parser.add_argument('--log', help='log directory for tensorboard')
    parser.add_argument('--workers', '-w', type=int, help='num of dataloader workers')
    parser.add_argument('--shuffle', '-s', action='store_true', help='use dataset shuffle')
    parser.add_argument('--pin', '-p', action='store_true', help='use pin memory')
    parser.add_argument('--drop', '-d', action='store_true', help='use drop last')
    parser.add_argument('--use_valid', '-v', action='store_true', help='use validation dataset')
    parser.add_argument('--use_log', '-l', action='store_true', help='use logger')

    # argument parsing
    args = parser.parse_args()
    training_epoch = args.epoch
    batch_size = args.batch
    learning_rate = args.rate
    checkpoint_dir = args.checkpoint
    interval = args.interval
    model_path = args.model
    gpu = args.gpu
    log_dir = args.log
    workers = args.workers
    shuffle = args.shuffle
    pin = args.pin
    drop = args.drop
    use_valid = args.use_valid
    use_log = args.use_log

    # configuration setting
    config = load(args.config)
    print('config : ', args.config)
    if training_epoch is not None:
        config.training_epoch = training_epoch
    if batch_size is not None:
        config.batch_size = batch_size
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir
    if interval is not None:
        config.checkpoint_interval = interval
    if model_path is not None:
        config.model_path = model_path
    if gpu is not None:
        config.gpu_index = gpu
    if log_dir is not None:
        config.log_dir = log_dir
    if workers is not None:
        config.num_workers = workers
    shuffle = args.shuffle
    pin = args.pin
    drop = args.drop
    config.use_valid = use_valid
    config.use_log = use_log

    device, sw_train, sw_valid = setup(config)

    train_loader, valid_loader = build_dataset(config)

    generator, mpd, msd, optim_g, optim_d, scheduler_g, scheduler_d = build_model(config, device)

    train_model(train_loader, valid_loader, generator, mpd, msd, optim_g, optim_d, scheduler_g, scheduler_d, device, sw_train, sw_valid, config, logger)
