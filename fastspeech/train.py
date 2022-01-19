import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import time
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
import preference
import tqdm
import pickle
from scaler import load_scaler
from text.korean import symbols
import numpy as np
import logging
import sys


def train_model(
        dataloader: DataLoader,
        valid_dataloader: DataLoader,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        schedular: ScheduledOptim,
        sw_train,
        sw_valid,
        config: preference.Config,
        logger,
        start_epoch=0):
    print('max epoch: ', config.training_epoch)
    start = time.time()
    for epoch in range(start_epoch, config.training_epoch):
        count = 0
        cum_total_loss = 0
        cum_mel_loss = 0
        cum_mel_postnet_loss = 0
        cum_duration_loss = 0
        cum_f0_loss = 0
        cum_energy_loss = 0
        for text, mel, d, log_d, f0, std_f0, phone_mask, mel_mask, max_mel_len in dataloader:
            text = text.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)
            log_d = log_d.to(device, non_blocking=True)
            f0 = f0.to(device, non_blocking=True)
            std_f0 = std_f0.to(device, non_blocking=True)
            phone_mask = phone_mask.to(device, non_blocking=True)
            mel_mask = mel_mask.to(device, non_blocking=True)
            max_mel_len = max_mel_len.to(device, non_blocking=True)

            # Forward
            mel_output, mel_postnet_output, src_mask, mel_mask = model(
                text, d, f0, std_f0, max_mel_len, phone_mask, mel_mask)

            # Cal Loss
            mel_loss, mel_postnet_loss = loss(mel_output, mel_postnet_output, mel, ~src_mask, ~mel_mask)
            sum_loss = mel_postnet_loss

            cum_total_loss += sum_loss.item() * text.shape[0]

            # Backward
            sum_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_thresh)

            # Update weights
            schedular.step_and_update_lr()
            schedular.zero_grad()

            count += text.shape[0]

        # end of epoch
        config.last_epoch = epoch
        config.step = schedular.n_current_steps

        # log
        sw_train.add_scalar('acoustic/loss', cum_total_loss / count, epoch)

        train_loss = cum_total_loss / count

        if valid_dataloader is not None:
            count = 0
            cum_total_loss = 0
            cum_mel_loss = 0
            cum_mel_postnet_loss = 0
            cum_duration_loss = 0
            cum_f0_loss = 0
            cum_energy_loss = 0
            model.eval()
            with torch.no_grad():
                for text, mel, d, log_d, f0, std_f0, phone_mask, mel_mask, max_mel_len in valid_dataloader:
                    text = text.to(device, non_blocking=True)
                    mel = mel.to(device, non_blocking=True)
                    d = d.to(device, non_blocking=True)
                    log_d = log_d.to(device, non_blocking=True)
                    f0 = f0.to(device, non_blocking=True)
                    std_f0 = std_f0.to(device, non_blocking=True)
                    phone_mask = phone_mask.to(device, non_blocking=True)
                    mel_mask = mel_mask.to(device, non_blocking=True)
                    max_mel_len = max_mel_len.to(device, non_blocking=True)

                    # Forward
                    mel_output, mel_postnet_output, src_mask, mel_mask = model(
                        text, d, f0, std_f0, max_mel_len, phone_mask, mel_mask)

                    # Cal Loss
                    mel_loss, mel_postnet_loss = loss(mel_output, mel_postnet_output, mel, ~src_mask, ~mel_mask)
                    sum_loss = mel_postnet_loss

                    cum_total_loss += sum_loss.item() * text.shape[0]

                    count += text.shape[0]
            model.train()

            # log
            sw_valid.add_scalar('acoustic/loss', cum_total_loss / count, epoch)
            valid_loss = cum_total_loss / count
            logger.info('epoch : {}, train loss {:4.3f} valid loss {:4.3f}'.format(epoch + 1, train_loss, valid_loss))
        else:
            print('epoch : {}, train loss {:4.3f}'.format(epoch + 1, train_loss))

        # check point
        end = time.time()
        if config.checkpoint_interval < (end - start):
            path = os.path.join(config.checkpoint_dir, '{}.tar'.format(epoch + 1))
            torch.save(
                {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()},
                path)
            config.last_checkpoint_file = path
            print('save checkpoint : ', path)

            path = os.path.join(config.checkpoint_dir, 'config.json')
            preference.save(config, path)
            print('save config : ', path)
            start = time.time()


def build_model(config: preference.Config, device: torch.device):
    model = FastSpeech2(
        config.decoder_hidden, config.n_mel_channels, ord('힣') - ord('가') + 3, config.max_seq_len, config.encoder_hidden,
        config.encoder_head, config.decoder_head, config.decoder_dropout, config.fft_filter_size,
        config.fft_kernel_size, config.encoder_dropout, config.log_offset, config.variance_predictor_filter_size,
        config.variance_predictor_kernel_size, config.variance_predictor_dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=config.betas, eps=config.eps,
                                 weight_decay=config.weight_decay)
    print('--Adam optimizer')
    print('betas: ', config.betas)
    print('eps:', config.eps)
    print('weight decay:', config.weight_decay)

    scheduler = ScheduledOptim(optimizer, config.decoder_hidden, config.n_warm_up_step, config.step)
    print('--learning rate schedule')
    print('decoder hidden: ', config.decoder_hidden)
    print('warm up step: ', config.n_warm_up_step)
    print('step: ', config.step)

    loss = FastSpeech2Loss().to(device)
    print('--fastspeech2 custom loss')

    print('{s:{c}^{n}}\n'.format(s='complete: model step', n=50, c='-'))

    return model, loss, optimizer, scheduler


def build_dataloader(files, mel_scaler, f0_scaler, energy_scaler, config: preference.Config, desc=''):
    txt_id_list = []
    mel_list = []
    d_list = []
    f0_list = []
    std_f0_list = []
    for file in tqdm.tqdm(files, desc=desc):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            txt_id_list.append(np.array(data['text_id']))
            d_list.append(np.array(data['duration']))
            f0_list.append(np.array(data['f0']))
            std_f0_list.append(f0_scaler.transform(np.array(data['f0']).reshape(-1, 1)).flatten())
            mel_list.append(mel_scaler.transform(np.array(data['mel']).T))
    dataset = Dataset(txt_id_list, mel_list, d_list, f0_list, std_f0_list, mel_scaler, f0_scaler,
                      energy_scaler, config.log_offset)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        collate_fn=dataset.collate_fn,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=config.persistent_workers)
    print('batch size: ', config.batch_size)
    print('num workers: ', config.num_workers)
    print('shuffle: ', config.shuffle)
    print('pin memory: ', config.pin_memory)
    print('drop last: ', config.drop_last)
    print('persistent workers: ', config.persistent_workers)

    return dataloader

def build_dataset(config: preference.Config):
    mel_scaler, f0_scaler, energy_scaler = load_scaler(config.mel_scaler, config.f0_scaler, config.energy_scaler)
    train_dataloader = build_dataloader(config.train_file, mel_scaler, f0_scaler, energy_scaler, config, 'load train dataset')

    if len(config.valid_file) > 0:
        valid_dataloader = build_dataloader(config.valid_file, mel_scaler, f0_scaler, energy_scaler, config, 'load valid datset')
    else:
        valid_dataloader = None

    print('batch size: ', config.batch_size)
    print('num workers: ', config.num_workers)
    print('shuffle: ', config.shuffle)
    print('pin memory: ', config.pin_memory)
    print('drop last: ', config.drop_last)
    print('persistent workers: ', config.persistent_workers)

    print('{s:{c}^{n}}\n'.format(s='complete: dataset step', n=50, c='-'))

    return train_dataloader, valid_dataloader


def setup(config: preference.Config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(config.gpu_index)

    if config.gpu_index >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print("checkpoint directory : ", config.checkpoint_dir)

    path = os.path.join(config.log, 'train')
    os.makedirs(path, exist_ok=True)
    print('train log directory: ', path)
    config.train_log = path

    path = os.path.join(config.log, 'valid')
    os.makedirs(path, exist_ok=True)
    print('valid log directory : ', path)
    config.valid_log = path

    sw_train = SummaryWriter(config.train_log)
    sw_valid = SummaryWriter(config.valid_log)

    print('{s:{c}^{n}}\n'.format(s='complete: setup step', n=50, c='-'))

    return device, sw_train, sw_valid


if __name__ == "__main__":
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
    parser.add_argument('--rate', '-r', type=float, help='learning rate for training')
    parser.add_argument('--checkpoint', '-c', help='directory for save checkpoint file')
    parser.add_argument('--interval', '-t', type=int, help='check point save interval time(sec)')
    parser.add_argument('--model', '-m', help='model path for fine tuning')
    parser.add_argument('--gpu', '-g', type=int, help='gpu device index')
    parser.add_argument('--log', '-l', help='log directory for tensorboard')
    parser.add_argument('--workers', '-w', type=int, help='num of dataloader workers')
    parser.add_argument('--shuffle', '-s', type=bool, help='dataset shuffle use')
    parser.add_argument('--pin', '-p', type=bool, help='dataloader use pin memory')
    parser.add_argument('--drop', '-d', type=bool, help='dataloader use drop last')

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

    # configuration setting
    config = preference.load(args.config)
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
        config.log = log_dir
    if workers is not None:
        config.num_workers = workers
        if workers is 0:
            config.persistent_workers = False
    if shuffle is not None:
        config.shuffle = shuffle
    if pin is not None:
        config.pin_memory = pin
    if drop is not None:
        config.drop_last

    # setup
    device, sw_train, sw_valid = setup(config)

    # dataset
    train_dataloader, valid_dataloader = build_dataset(config)

    # model
    model, loss, optimizer, schedular = build_model(config, device)

    # train
    train_model(train_dataloader, valid_dataloader, model, loss, optimizer, schedular, sw_train, sw_valid, config, logger)
