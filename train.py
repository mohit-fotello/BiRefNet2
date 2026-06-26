import os
import datetime
from contextlib import nullcontext
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
if tuple(map(int, torch.__version__.split('+')[0].split(".")[:3])) >= (2, 5, 0):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from config import Config
from loss import PixLoss, ClsLoss
from dataset import MyData
from models.birefnet import BiRefNet
from utils import Logger, AverageMeter, set_seed, check_state_dict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    if value.lower() in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


WANDB_PROJECT = 'birefnet-wallmasking'
WANDB_ENTITY = None
WANDB_RUN_NAME = None
WANDB_MODE = 'online'
WANDB_LOG_FREQ = 1
WANDB_LOG_SAMPLES = True
WANDB_SAMPLE_FREQ = 10
WANDB_VAL_SAMPLE_FREQ = 1
WANDB_NUM_SAMPLES = 2
WANDB_LOG_MODEL = False
WANDB_LOG_BEST_MODEL = True


parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=250, type=int)
parser.add_argument('--ckpt_dir', default='ckpts/tmp', help='Temporary folder')
parser.add_argument('--batch_size', default=None, type=int, help='Override Config.batch_size.')
parser.add_argument('--val_sets', default=None, type=str, help='Override Config.testsets for validation, e.g. val or val+test.')
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', action='store_true', help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
parser.add_argument('--wandb', default=True, type=str2bool, nargs='?', const=True, help='Enable Weights & Biases logging.')
args = parser.parse_args()

config = Config()
if args.val_sets is not None:
    val_sets = args.val_sets.replace('+', ',')
    val_set_names = set(val_sets.split(','))
    config.testsets = val_sets
    config.training_set = '+'.join(ds for ds in config.training_set.split('+') if ds not in val_set_names)
if args.batch_size is not None:
    config.batch_size = args.batch_size
    config.num_workers = max(4, config.batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_seed = 0
wandb = None
wandb_run = None

if args.use_accelerate:
    from accelerate import Accelerator, utils
    mixed_precision = config.mixed_precision
    kwargs_handlers = [
            utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600*10)),
            utils.DistributedDataParallelKwargs(find_unused_parameters=False),
            utils.GradScalerKwargs(backoff_factor=0.5),
    ]
    if mixed_precision == 'fp8':
        kwargs_handlers.append(utils.AORecipeKwargs())
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        kwargs_handlers=kwargs_handlers,
    )
    accelerator.print(accelerator.state)
    accelerator.print('backbone:', config.bb, ', freeze_bb:', config.freeze_bb)
    args.dist = False

# DDP
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
    device_seed = device
else:
    if args.use_accelerate:
        device = accelerator.local_process_index
        device_seed = device

if config.rand_seed:
    set_seed(config.rand_seed + device_seed)

epoch_st = 1
# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_idx = 1

# log model and optimizer params
# logger.info("Model details:"); logger.info(model)
# if args.use_accelerate and accelerator.mixed_precision != 'no':
#     config.compile = False
logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:"); logger.info(args)
print('batch size:', config.batch_size)

from dataset import custom_collate_fn


def is_main_process():
    if args.use_accelerate:
        return accelerator.is_main_process
    if to_be_distributed:
        return int(os.environ.get('RANK', '0')) == 0
    return True


def is_wandb_enabled():
    return wandb_run is not None and is_main_process()


def config_to_dict():
    simple_types = (str, int, float, bool, type(None), tuple, list, dict)
    return {k: v for k, v in config.__dict__.items() if isinstance(v, simple_types)}


def init_wandb():
    global wandb, wandb_run
    if not args.wandb or not is_main_process():
        return
    try:
        import wandb as wandb_module
    except ImportError as exc:
        raise SystemExit('Install wandb or run without --wandb. Try: pip install wandb') from exc

    wandb = wandb_module
    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        mode=WANDB_MODE,
        dir=args.ckpt_dir,
        config={
            'args': vars(args),
            'config': config_to_dict(),
        },
    )
    wandb.define_metric('train/step')
    wandb.define_metric('train/*', step_metric='train/step')
    wandb.define_metric('epoch')
    wandb.define_metric('epoch/*', step_metric='epoch')


def tensor_to_wandb_image(tensor, image_type='mask'):
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        array = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    else:
        array = tensor.squeeze().clamp(0, 1).numpy()
    array = (array * 255).astype(np.uint8)
    return wandb.Image(array, mode='RGB' if image_type == 'rgb' else 'L')


def log_wandb_samples(inputs, gts, scaled_preds, epoch, batch_idx, global_step, prefix='train'):
    if not is_wandb_enabled() or not WANDB_LOG_SAMPLES:
        return
    pred = scaled_preds[-1]
    if pred.shape[2:] != gts.shape[2:]:
        pred = nn.functional.interpolate(pred, size=gts.shape[2:], mode='bilinear', align_corners=True)
    pred = pred.sigmoid()
    sample_count = min(WANDB_NUM_SAMPLES, inputs.shape[0])
    table = wandb.Table(columns=['epoch', 'batch', 'sample', 'image', 'ground_truth', 'prediction'])
    for sample_idx in range(sample_count):
        table.add_data(
            epoch,
            batch_idx,
            sample_idx,
            tensor_to_wandb_image(inputs[sample_idx], 'rgb'),
            tensor_to_wandb_image(gts[sample_idx], 'mask'),
            tensor_to_wandb_image(pred[sample_idx], 'mask'),
        )
    wandb.log({f'{prefix}/samples': table, 'train/step': global_step, 'epoch': epoch}, step=global_step)


def log_wandb_checkpoint(checkpoint_path, epoch):
    if not is_wandb_enabled() or not WANDB_LOG_MODEL:
        return
    artifact = wandb.Artifact(f'{wandb_run.name}-checkpoint', type='model')
    artifact.add_file(checkpoint_path)
    artifact.metadata = {'epoch': epoch, 'checkpoint_path': checkpoint_path}
    wandb_run.log_artifact(artifact, aliases=[f'epoch-{epoch}', 'latest'])


def log_wandb_best_checkpoints(best_checkpoints):
    if not is_wandb_enabled() or not WANDB_LOG_BEST_MODEL:
        return
    for rank, checkpoint in enumerate(sorted(best_checkpoints, key=lambda item: item['val_loss']), start=1):
        artifact = wandb.Artifact(f'{wandb_run.name}-best-{rank}', type='model')
        artifact.add_file(checkpoint['path'])
        artifact.metadata = {
            'epoch': checkpoint['epoch'],
            'val_loss': checkpoint['val_loss'],
            'checkpoint_path': checkpoint['path'],
            'rank': rank,
        }
        wandb_run.log_artifact(artifact, aliases=[f'best-{rank}', f"epoch-{checkpoint['epoch']}"])


init_wandb()

def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    # Prepare dataloaders
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=DistributedSampler(dataset), drop_last=is_train, collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=is_train, sampler=None, drop_last=is_train, collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )


def init_data_loaders(to_be_distributed):
    # Prepare datasets
    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, data_size=None if config.dynamic_size else config.size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(config.training_set))
    val_sets = config.testsets.replace(',', '+')
    val_loader = None
    if val_sets:
        val_loader = prepare_dataloader(
            MyData(datasets=val_sets, data_size=None, is_train=False),
            config.batch_size_valid, to_be_distributed=to_be_distributed, is_train=False
        )
        print(len(val_loader), "batches of val dataloader {} have been created.".format(val_sets))
    else:
        logger.info('No validation set configured. Best checkpoints and validation samples will be skipped.')
    return train_loader, val_loader


def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    else:
        print('Undefined model: {}.'.format(config.model))
        return None
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global epoch_st
            epoch_st = int(args.resume.rstrip('.pth').split('epoch_')[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)
    if config.compile:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=[p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config.lr_min
    )
    # logger.info("Optimizer details:"); logger.info(optimizer)

    return model, optimizer, lr_scheduler


class Trainer:
    def __init__(
        self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader, self.val_loader = data_loaders
        if args.use_accelerate:
            if self.val_loader is None:
                self.train_loader, self.model, self.optimizer = accelerator.prepare(self.train_loader, self.model, self.optimizer)
            else:
                self.train_loader, self.val_loader, self.model, self.optimizer = accelerator.prepare(
                    self.train_loader, self.val_loader, self.model, self.optimizer
                )
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()
        
        # Others
        self.loss_log = AverageMeter()
        self.val_loss_log = AverageMeter()
        self.best_checkpoints = []
        self.global_step = 0

    def _batch_inputs_gts(self, batch):
        if args.use_accelerate:
            return batch[0], batch[1]
        return batch[0].to(device), batch[1].to(device)

    def _train_batch(self, batch, epoch, batch_idx, capture_samples=False):
        inputs, gts = self._batch_inputs_gts(batch)
        class_labels = batch[2] if args.use_accelerate else batch[2].to(device)
        self.optimizer.zero_grad()
        scaled_preds, class_preds_lst = self.model(inputs)
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
            # self.loss_dict['loss_gdt'] = loss_gdt.item()
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels)
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Loss
        loss_pix, loss_dict_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)
        self.loss_dict.update(loss_dict_pix)
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0

        loss_value = loss.item()
        self.loss_log.update(loss_value, inputs.size(0))
        if capture_samples:
            log_wandb_samples(inputs, gts, scaled_preds, epoch, batch_idx, self.global_step)
        if args.use_accelerate:
            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)
        else:
            loss.backward()
        self.optimizer.step()
        return loss_value

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_log.reset()
        self.loss_dict = {}
        if config.task != 'WallMasking' and epoch > args.epochs + config.finetune_last_epochs:
            if config.task == 'Matting':
                for loss_name, multiplier in {'mae': 1, 'mse': 0.9, 'ssim': 0.9}.items():
                    if loss_name in self.pix_loss.lambdas_pix_last:
                        self.pix_loss.lambdas_pix_last[loss_name] *= multiplier
            else:
                for loss_name, multiplier in {'bce': 0, 'ssim': 1, 'iou': 0.5, 'mae': 0.9}.items():
                    if loss_name in self.pix_loss.lambdas_pix_last:
                        self.pix_loss.lambdas_pix_last[loss_name] *= multiplier

        for batch_idx, batch in enumerate(self.train_loader):
            # with nullcontext if not args.use_accelerate or accelerator.gradient_accumulation_steps <= 1 else accelerator.accumulate(self.model):
            capture_samples = (
                WANDB_LOG_SAMPLES
                and batch_idx == 0
                and epoch % max(WANDB_SAMPLE_FREQ, 1) == 0
            )
            loss_value = self._train_batch(batch, epoch, batch_idx, capture_samples=capture_samples)
            if is_wandb_enabled() and self.global_step % max(WANDB_LOG_FREQ, 1) == 0:
                wandb_logs = {
                    'train/step': self.global_step,
                    'train/loss': loss_value,
                    'train/loss_avg': self.loss_log.avg,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'batch': batch_idx,
                }
                wandb_logs.update({f'train/{name}': value for name, value in self.loss_dict.items()})
                wandb.log(wandb_logs, step=self.global_step)
            # Logger
            if (epoch < 2 and batch_idx < 100 and batch_idx % 20 == 0) or batch_idx % max(100, len(self.train_loader) / 100 // 100 * 100) == 0:
                info_progress = f'Epoch[{epoch}/{args.epochs}] Iter[{batch_idx}/{len(self.train_loader)}].'
                info_loss = 'Training Losses:'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += f' {loss_name}: {loss_value:.5g} |'
                logger.info(' '.join((info_progress, info_loss)))
            self.global_step += 1
        info_loss = f'@==Final== Epoch[{epoch}/{args.epochs}]  Training Loss: {self.loss_log.avg:.5g}  '
        logger.info(info_loss)
        if is_wandb_enabled():
            wandb.log(
                {
                    'epoch': epoch,
                    'epoch/train_loss': self.loss_log.avg,
                    'epoch/lr': self.optimizer.param_groups[0]['lr'],
                },
                step=self.global_step,
            )

        self.lr_scheduler.step()
        return self.loss_log.avg

    def validate_epoch(self, epoch):
        if self.val_loader is None:
            return None
        self.model.eval()
        self.val_loss_log.reset()
        val_loss_dict = {}
        capture_samples = epoch % max(WANDB_VAL_SAMPLE_FREQ, 1) == 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                inputs, gts = self._batch_inputs_gts(batch)
                scaled_preds = self.model(inputs)
                loss_pix, loss_dict_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1), pix_loss_lambda=1.0)
                val_loss_dict = loss_dict_pix
                loss_value = loss_pix.item()
                self.val_loss_log.update(loss_value, inputs.size(0))
                if is_wandb_enabled() and batch_idx == 0 and capture_samples:
                    log_wandb_samples(inputs, gts, scaled_preds, epoch, batch_idx, self.global_step, prefix='val')

        logger.info(f'@==Final== Epoch[{epoch}/{args.epochs}]  Validation Loss: {self.val_loss_log.avg:.5g}  ')
        if is_wandb_enabled():
            wandb_logs = {
                'epoch': epoch,
                'epoch/val_loss': self.val_loss_log.avg,
            }
            wandb_logs.update({f'epoch/val_{name}': value for name, value in val_loss_dict.items()})
            wandb.log(wandb_logs, step=self.global_step)
        return self.val_loss_log.avg

    def model_state_dict(self):
        if args.use_accelerate:
            return accelerator.unwrap_model(self.model).state_dict()
        return self.model.module.state_dict() if to_be_distributed else self.model.state_dict()

    def save_best_checkpoint(self, epoch, val_loss):
        if val_loss is None:
            return
        worst_best = max((checkpoint['val_loss'] for checkpoint in self.best_checkpoints), default=None)
        if len(self.best_checkpoints) >= 3 and val_loss >= worst_best:
            return

        checkpoint_path = os.path.join(args.ckpt_dir, f'best_epoch_{epoch:04d}_val_{val_loss:.6f}.pth')
        torch.save(self.model_state_dict(), checkpoint_path)
        self.best_checkpoints.append({'epoch': epoch, 'val_loss': val_loss, 'path': checkpoint_path})
        self.best_checkpoints.sort(key=lambda item: item['val_loss'])
        while len(self.best_checkpoints) > 3:
            removed = self.best_checkpoints.pop()
            if os.path.exists(removed['path']):
                os.remove(removed['path'])
        best_summary = ', '.join(f"epoch {item['epoch']}: {item['val_loss']:.5g}" for item in self.best_checkpoints)
        logger.info(f'Best validation checkpoints: {best_summary}')


def main():

    trainer = Trainer(
        data_loaders=init_data_loaders(to_be_distributed),
        model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
    )

    for epoch in range(epoch_st, args.epochs+1):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate_epoch(epoch)
        trainer.save_best_checkpoint(epoch, val_loss)
        # Save only the latest checkpoint to avoid accumulating large epoch files.
        if epoch % config.save_step == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.ckpt_dir, 'latest.pth')
            torch.save(trainer.model_state_dict(), checkpoint_path)
            log_wandb_checkpoint(checkpoint_path, epoch)
    log_wandb_best_checkpoints(trainer.best_checkpoints)
    if to_be_distributed:
        destroy_process_group()
    if is_wandb_enabled():
        wandb.finish()


if __name__ == '__main__':
    main()
