# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import math
import torch
import torch.nn.functional as F
import einops
from accelerate import Accelerator
from accelerate.state import PartialState
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from collections import defaultdict
from torchvision.transforms import v2

from models.mast import build as build_mast
from runtime_option import runtime_option
from utils.misc import yaml_to_dict, set_seed
from configs.util import load_super_config, update_config
from log.logger import Logger
from data import build_dataset
from data.naive_sampler import NaiveSampler
from data.util import collate_fn
from log.log import TPS, Metrics
from models.misc import load_detr_pretrain, save_checkpoint, load_checkpoint
from models.misc import get_model
from utils.nested_tensor import NestedTensor
from utils.train_utils import batch_iterator, tensor_dict_index_select
from submit_and_evaluate import submit_and_evaluate_one_model


def train_engine(config: dict):
    # Init some settings:
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    # Init Accelerator at beginning:
    accelerator = Accelerator()
    state = PartialState()
    # Also, we set the seed:
    set_seed(config["SEED"])
    # Set the sharing strategy (to avoid error: too many open files):
    torch.multiprocessing.set_sharing_strategy('file_system')   # if not, raise error: too many open files.

    # Init Logger:
    logger = Logger(
        logdir=os.path.join(outputs_dir, "train"),
        use_wandb=config["USE_WANDB"],
        config=config,
        exp_owner=config["EXP_OWNER"],
        exp_project=config["EXP_PROJECT"],
        exp_group=config["EXP_GROUP"],
        exp_name=config["EXP_NAME"],
    )
    logger.info(f"We init the logger at {logger.logdir}.")
    if config["USE_WANDB"] is False:
        logger.warning("The wandb is not used in this experiment.")
    logger.info(f"The distributed type is {state.distributed_type}.")
    logger.config(config=config)

    # Build training dataset:
    train_dataset = build_dataset(config=config)
    logger.dataset(train_dataset)
    # Build training data sampler:
    if "DATASET_WEIGHTS" in config:
        data_weights = defaultdict(lambda: defaultdict())
        for _ in range(len(config["DATASET_WEIGHTS"])):
            data_weights[config["DATASETS"][_]][config["DATASET_SPLITS"][_]] = config["DATASET_WEIGHTS"][_]
        data_weights = dict(data_weights)
    else:
        data_weights = None
    train_sampler = NaiveSampler(
        data_source=train_dataset,
        sample_steps=config["SAMPLE_STEPS"],
        sample_lengths=config["SAMPLE_LENGTHS"],
        sample_intervals=config["SAMPLE_INTERVALS"],
        length_per_iteration=config["LENGTH_PER_ITERATION"],
        data_weights=data_weights,
    )
    # Build training data loader:
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS"],
        prefetch_factor=config["PREFETCH_FACTOR"] if config["NUM_WORKERS"] > 0 else None,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Init the training states:
    train_states = {
        "start_epoch": 0,
        "global_step": 0
    }

    # Build MAST model:
    model, detr_criterion = build_mast(config=config)
    # Load the pre-trained DETR:
    load_detr_pretrain(
        model=model, pretrain_path=config["DETR_PRETRAIN"], num_classes=config["NUM_CLASSES"],
        default_class_idx=config["DETR_DEFAULT_CLASS_IDX"] if "DETR_DEFAULT_CLASS_IDX" in config else None,
    )
    logger.success(
        log=f"Load the pre-trained DETR from '{config['DETR_PRETRAIN']}'. "
    )

    # Build Optimizer:
    if config["DETR_NUM_TRAIN_FRAMES"] == 0:
        for n, p in model.named_parameters():
            if "detr" in n:
                p.requires_grad = False     # only train the MAST part.
    param_groups = get_param_groups(model, config)
    optimizer = AdamW(
        params=param_groups,
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
    )
    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=config["SCHEDULER_MILESTONES"],
        gamma=config["SCHEDULER_GAMMA"],
    )

    # Other infos:
    only_detr = config["ONLY_DETR"]

    # Resuming:
    if config["RESUME_MODEL"] is not None:
        load_checkpoint(
            model=model,
            path=config["RESUME_MODEL"],
            optimizer=optimizer if config["RESUME_OPTIMIZER"] else None,
            scheduler=scheduler if config["RESUME_SCHEDULER"] else None,
            states=train_states,
        )
        # Different processing on scheduler:
        if config["RESUME_SCHEDULER"]:
            scheduler.step()
        else:
            for _ in range(0, train_states["start_epoch"]):
                scheduler.step()
        logger.success(
            log=f"Resume the model from '{config['RESUME_MODEL']}', "
                f"optimizer={config['RESUME_OPTIMIZER']}, "
                f"scheduler={config['RESUME_SCHEDULER']}, "
                f"states={train_states}. "
                f"Start from epoch {train_states['start_epoch']}, step {train_states['global_step']}."
        )

    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer,
        # device_placement=[False]        # whether to place the data on the device
    )

    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        logger.info(log=f"Start training epoch {epoch}.")
        epoch_start_timestamp = TPS.timestamp()
        # Prepare the sampler for the current epoch:
        train_sampler.prepare_for_epoch(epoch=epoch)
        # Train one epoch:
        train_metrics = train_one_epoch(
            accelerator=accelerator,
            logger=logger,
            states=train_states,
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            detr_criterion=detr_criterion,
            optimizer=optimizer,
            only_detr=only_detr,
            lr_warmup_epochs=config["LR_WARMUP_EPOCHS"],
            lr_warmup_tgt_lr=config["LR"],
            detr_num_train_frames=config["DETR_NUM_TRAIN_FRAMES"],
            detr_num_checkpoint_frames=config["DETR_NUM_CHECKPOINT_FRAMES"],
            detr_criterion_batch_len=config.get("DETR_CRITERION_BATCH_LEN", 10),
            assign_loss_weight=config.get("ASSIGN_LOSS_WEIGHT", 1.0),
            accumulate_steps=config["ACCUMULATE_STEPS"],
            separate_clip_norm=config.get("SEPARATE_CLIP_NORM", True),
            max_clip_norm=config.get("MAX_CLIP_NORM", 0.1),
            use_accelerate_clip_norm=config.get("USE_ACCELERATE_CLIP_NORM", True),
            # For multi last checkpoints:
            outputs_dir=outputs_dir,
            is_last_epochs=(epoch == config["EPOCHS"] - 1),
            multi_last_checkpoints=config["MULTI_LAST_CHECKPOINTS"],
        )

        # Get learning rate:
        lr = optimizer.state_dict()["param_groups"][-1]["lr"]
        train_metrics["lr"].update(lr)
        train_metrics["lr"].sync()
        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
        logger.metrics(
            log=f"[Finish epoch: {epoch}] [Time: {time_per_epoch}] ",
            metrics=train_metrics,
            fmt="{global_average:.4f}",
            statistic="global_average",
            global_step=train_states["global_step"],
            prefix="epoch",
            x_axis_step=epoch,
            x_axis_name="epoch",
        )

        # Save checkpoint:
        if (epoch + 1) % config["SAVE_CHECKPOINT_PER_EPOCH"] == 0:
            save_checkpoint(
                model=model,
                path=os.path.join(outputs_dir, f"checkpoint_{epoch}.pth"),
                states=train_states,
                optimizer=optimizer,
                scheduler=scheduler,
                only_detr=only_detr,
            )
            if config["INFERENCE_DATASET"] is not None:
                assert config["INFERENCE_SPLIT"] is not None, f"Please set the INFERENCE_SPLIT for inference."
                eval_metrics = submit_and_evaluate_one_model(
                    is_evaluate=True,
                    accelerator=accelerator,
                    state=state,
                    logger=logger,
                    model=model,
                    data_root=config["DATA_ROOT"],
                    dataset=config["INFERENCE_DATASET"],
                    data_split=config["INFERENCE_SPLIT"],
                    outputs_dir=os.path.join(outputs_dir, "train", "eval_during_train", f"epoch_{epoch}"),
                    image_max_longer=config["INFERENCE_MAX_LONGER"],
                    size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
                    miss_tolerance=config["MISS_TOLERANCE"],
                    use_sigmoid=config["USE_FOCAL_LOSS"] if "USE_FOCAL_LOSS" in config else False,
                    assignment_protocol=config["ASSIGNMENT_PROTOCOL"] if "ASSIGNMENT_PROTOCOL" in config else "hungarian",
                    det_thresh=config["DET_THRESH"],
                    newborn_thresh=config["NEWBORN_THRESH"],
                    id_thresh=config["ID_THRESH"],
                    area_thresh=config["AREA_THRESH"],
                    inference_only_detr=config["INFERENCE_ONLY_DETR"] if config["INFERENCE_ONLY_DETR"] is not None
                    else config["ONLY_DETR"],
                )
                eval_metrics.sync()
                logger.metrics(
                    log=f"[Eval epoch: {epoch}] ",
                    metrics=eval_metrics,
                    fmt="{global_average:.4f}",
                    statistic="global_average",
                    global_step=train_states["global_step"],
                    prefix="epoch",
                    x_axis_step=epoch,
                    x_axis_name="epoch",
                )

        logger.success(log=f"Finish training epoch {epoch}.")
        # Prepare for next step:
        scheduler.step()
    pass


def train_one_epoch(
        # Infos:
        accelerator: Accelerator,
        logger: Logger,
        states: dict,
        epoch: int,
        dataloader: DataLoader,
        model,
        detr_criterion,
        optimizer,
        only_detr,
        lr_warmup_epochs: int,
        lr_warmup_tgt_lr: float,
        detr_num_train_frames: int,
        detr_num_checkpoint_frames: int,
        detr_criterion_batch_len: int,
        assign_loss_weight: float = 1.0,
        accumulate_steps: int = 1,
        separate_clip_norm: bool = True,
        max_clip_norm: float = 0.1,
        use_accelerate_clip_norm: bool = True,
        logging_interval: int = 20,
        # For multi last checkpoints:
        outputs_dir: str = None,
        is_last_epochs: bool = False,
        multi_last_checkpoints: int = 0,
):
    current_last_checkpoint_idx = 0

    model.train()
    tps = TPS()     # time per step
    metrics = Metrics()
    optimizer.zero_grad()
    step_timestamp = tps.timestamp()
    device = accelerator.device
    _B = dataloader.batch_sampler.batch_size
    _num_gts_per_frame = 0

    # Prepare for gradient clip norm:
    model_without_ddp = get_model(model)
    detr_params = []
    other_params = []
    for name, param in model_without_ddp.named_parameters():
        if "detr" in name:
            detr_params.append(param)
        else:
            other_params.append(param)

    for step, samples in enumerate(dataloader):
        images, annotations, metas = samples["images"], samples["annotations"], samples["metas"]
        # Normalize the images:
        # (Normally, it should be done in the dataloader, but here we do it in the training loop (on cuda).)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        images.tensors = v2.functional.to_dtype(images.tensors, dtype=torch.float32, scale=True)
        images.tensors = v2.functional.normalize(images.tensors, mean=mean, std=std)
        # A hack implementation to recover 0.0 in the masked regions:
        images.tensors = images.tensors * (~images.mask[:, :, None, ...]).to(torch.float32)
        images.tensors = images.tensors.contiguous()

        # Learning rate warmup:
        if epoch < lr_warmup_epochs:
            # Do warmup:
            lr_warmup(
                optimizer=optimizer,
                epoch=epoch, curr_iter=step, tgt_lr=lr_warmup_tgt_lr,
                warmup_epochs=lr_warmup_epochs, num_iter_per_epoch=len(dataloader),
            )

        _B, _T = len(annotations), len(annotations[0])
        detr_num_train_frames_ = min(detr_num_train_frames, _T)

        # Which frames get gradients for the DETR backbone+encoder
        random_frame_idxs = torch.randperm(_T, device=device)
        detr_train_frame_idxs = set(random_frame_idxs[:detr_num_train_frames_].tolist())

        if only_detr:
            # ── Pretrain mode: standard DETR forward, all frames flattened ──────
            detr_targets_flatten = annotations_to_flatten_detr_targets(
                annotations=annotations, device=device
            )
            # Flatten images (B, T, C, H, W) → (B*T, C, H, W)
            flat_tensors = einops.rearrange(images.tensors, "b t c h w -> (b t) c h w").contiguous()
            flat_mask = einops.rearrange(images.mask, "b t h w -> (b t) h w").contiguous()
            flat_images = NestedTensor(flat_tensors, flat_mask)

            detr_outputs = model(flat_images)
            detr_loss_dict, _ = detr_criterion(
                outputs=detr_outputs, targets=detr_targets_flatten,
                batch_len=detr_criterion_batch_len,
            )

            with accelerator.autocast():
                detr_weight_dict = detr_criterion.weight_dict
                loss = sum(
                    detr_loss_dict[k] * detr_weight_dict[k]
                    for k in detr_loss_dict if k in detr_weight_dict
                )
                metrics.update(name="loss", value=loss.item())
                metrics.update(name="detr_loss", value=loss.item())
                for k, v in detr_loss_dict.items():
                    metrics.update(name=k, value=v.item())
                loss /= accumulate_steps

            accelerator.backward(loss)
            if (step + 1) % accumulate_steps == 0:
                if use_accelerate_clip_norm:
                    if separate_clip_norm:
                        detr_grad_norm = accelerator.clip_grad_norm_(detr_params, max_norm=max_clip_norm)
                        other_grad_norm = accelerator.clip_grad_norm_(other_params, max_norm=max_clip_norm)
                    else:
                        detr_grad_norm = other_grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), max_norm=max_clip_norm)
                else:
                    if separate_clip_norm:
                        accelerator.unscale_gradients()
                        detr_grad_norm = torch.nn.utils.clip_grad_norm_(detr_params, max_clip_norm)
                        other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, max_clip_norm)
                    else:
                        accelerator.unscale_gradients()
                        detr_grad_norm = other_grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_clip_norm)
                metrics.update(name="detr_grad_norm", value=detr_grad_norm.item())
                metrics.update(name="other_grad_norm", value=other_grad_norm.item())
                optimizer.step()
                optimizer.zero_grad()

        else:
            # ── Tracking mode: frame-by-frame MAST loop ──────────────────────────
            # Phase 1: Run backbone+encoder for all T frames
            model_unwrapped = get_model(model)
            all_encoder_outputs = []
            for t in range(_T):
                frame_tensors = images.tensors[:, t].contiguous()   # [B, C, H, W]
                frame_mask = images.mask[:, t].contiguous()          # [B, H, W]
                frame_nt = NestedTensor(frame_tensors, frame_mask)
                if t in detr_train_frame_idxs:
                    enc_out = model_unwrapped.forward_encoder(frame_nt)
                else:
                    with torch.no_grad():
                        enc_out = model_unwrapped.forward_encoder(frame_nt)
                all_encoder_outputs.append(enc_out)

            # Phase 2: Frame-by-frame MAST decoder + losses
            track_queries = None        # [B, N_track, d_model] or None
            track_spatial_info = None   # [B, N_track, d_model] or None
            track_gt_ids = None         # list of lists: [[id, ...], ...] per batch

            total_detect_loss = torch.tensor(0.0, device=device)
            total_assign_loss = torch.tensor(0.0, device=device)
            num_assign_frames = 0

            for t in range(_T):
                detr_targets_t = [
                    {
                        "boxes": annotations[b][t]["bbox"].to(device),
                        "labels": annotations[b][t]["category"].to(device),
                    }
                    for b in range(_B)
                ]

                mast_out = model_unwrapped.forward_decoder(
                    encoder_output=all_encoder_outputs[t],
                    track_queries=track_queries,
                    track_spatial_info=track_spatial_info,
                )

                detect_loss_dict, hungarian_indices_t = detr_criterion(
                    outputs=mast_out, targets=detr_targets_t,
                    batch_len=detr_criterion_batch_len,
                )
                detr_weight_dict = detr_criterion.weight_dict
                detect_loss_t = sum(
                    detect_loss_dict[k] * detr_weight_dict[k]
                    for k in detect_loss_dict if k in detr_weight_dict
                )
                total_detect_loss = total_detect_loss + detect_loss_t

                # Assignment loss (frame 2 onward)
                if track_queries is not None and track_queries.shape[1] > 0:
                    gt_assignment = build_gt_assignment(
                        track_gt_ids=track_gt_ids,
                        hungarian_indices=hungarian_indices_t,
                        annotations_t=[annotations[b][t] for b in range(_B)],
                        num_detect=model_unwrapped.num_queries,
                        device=device,
                    )
                    assignment_logits = mast_out["assignment_logits"]  # [B, N_track, N_detect+1]
                    assign_loss_t = F.cross_entropy(
                        assignment_logits.reshape(-1, assignment_logits.shape[-1]),
                        gt_assignment.reshape(-1),
                        reduction='mean',
                    )
                    total_assign_loss = total_assign_loss + assign_loss_t
                    num_assign_frames += 1
                else:
                    gt_assignment = None

                # Teacher forcing update for next frame
                track_queries, track_spatial_info, track_gt_ids = teacher_forcing_update(
                    model=model_unwrapped,
                    mast_out=mast_out,
                    gt_assignment=gt_assignment,
                    hungarian_indices=hungarian_indices_t,
                    annotations_t=[annotations[b][t] for b in range(_B)],
                    prev_track_queries=track_queries,
                    prev_track_gt_ids=track_gt_ids,
                    device=device,
                )

            detect_loss = total_detect_loss / _T
            assign_loss = total_assign_loss / max(num_assign_frames, 1)
            loss = detect_loss + assign_loss_weight * assign_loss

            with accelerator.autocast():
                metrics.update(name="loss", value=loss.item())
                metrics.update(name="detr_loss", value=detect_loss.item())
                metrics.update(name="assign_loss", value=assign_loss.item())
                loss /= accumulate_steps

            accelerator.backward(loss)
            if (step + 1) % accumulate_steps == 0:
                if use_accelerate_clip_norm:
                    if separate_clip_norm:
                        detr_grad_norm = accelerator.clip_grad_norm_(detr_params, max_norm=max_clip_norm)
                        other_grad_norm = accelerator.clip_grad_norm_(other_params, max_norm=max_clip_norm)
                    else:
                        detr_grad_norm = other_grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), max_norm=max_clip_norm)
                else:
                    if separate_clip_norm:
                        accelerator.unscale_gradients()
                        detr_grad_norm = torch.nn.utils.clip_grad_norm_(detr_params, max_clip_norm)
                        other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, max_clip_norm)
                    else:
                        accelerator.unscale_gradients()
                        detr_grad_norm = other_grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_clip_norm)
                metrics.update(name="detr_grad_norm", value=detr_grad_norm.item())
                metrics.update(name="other_grad_norm", value=other_grad_norm.item())
                optimizer.step()
                optimizer.zero_grad()

        # Logging:
        tps.update(tps=tps.timestamp() - step_timestamp)
        step_timestamp = tps.timestamp()
        # Logging:
        if step % logging_interval == 0:
            # logger.info(f"[Epoch: {epoch}] [{step}/{total_steps}] [tps: {tps.average:.2f}s]")
            # Get learning rate for current step:
            _lr = optimizer.state_dict()["param_groups"][-1]["lr"]
            # Get the GPU memory usage:
            torch.cuda.synchronize()
            _cuda_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            _cuda_memory = torch.tensor([_cuda_memory], device=device)
            # _cuda_memory_reduce = accelerator.reduce(_cuda_memory, reduction="none")
            _gathered_cuda_memory = accelerator.gather(_cuda_memory)
            _max_cuda_memory = _gathered_cuda_memory.max().item()
            accelerator.wait_for_everyone()
            # Clear some values:
            metrics["lr"].clear()  # clear the learning rate value from last step
            metrics["max_cuda_mem(MB)"].clear()
            # Update them to the metrics:
            metrics.update(name="lr", value=_lr)
            metrics.update(name="max_cuda_mem(MB)", value=_max_cuda_memory)
            # Sync the metrics:
            metrics.sync()
            eta = tps.eta(total_steps=len(dataloader), current_steps=step)
            logger.metrics(
                log=f"[Epoch: {epoch}] [{step}/{len(dataloader)}] "
                    f"[tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] ",
                metrics=metrics,
                global_step=states["global_step"],
            )
        # For multi last checkpoints:
        if is_last_epochs and multi_last_checkpoints > 0:
            if (step + 1) == int(math.ceil((len(dataloader) / multi_last_checkpoints) * (current_last_checkpoint_idx + 1))):
                _dir = os.path.join(outputs_dir, "multi_last_checkpoints")
                os.makedirs(_dir, exist_ok=True)
                save_checkpoint(
                    model=model,
                    path=os.path.join(_dir, f"last_checkpoint_{current_last_checkpoint_idx}.pth"),
                    states=states,
                    optimizer=None,
                    scheduler=None,
                    only_detr=only_detr,
                )
                logger.info(
                    log=f"Save the last checkpoint {current_last_checkpoint_idx} at step {step}."
                )
                current_last_checkpoint_idx += 1
        # Update the counters:
        states["global_step"] += 1
    states["start_epoch"] += 1
    return metrics


def get_param_groups(model, config) -> list[dict]:
    def _match_names(_name, _key_names):
        for _k in _key_names:
            if _k in _name:
                return True
        return False

    # Keywords:
    backbone_names = config["LR_BACKBONE_NAMES"]
    linear_proj_names = config["LR_LINEAR_PROJ_NAMES"]
    dictionary_names = config["LR_DICTIONARY_NAMES"]
    pass
    # Param groups:
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, backbone_names) and p.requires_grad],
            "lr_scale": config["LR_BACKBONE_SCALE"],
            "lr": config["LR"] * config["LR_BACKBONE_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, linear_proj_names) and p.requires_grad],
            "lr_scale": config["LR_LINEAR_PROJ_SCALE"],
            "lr": config["LR"] * config["LR_LINEAR_PROJ_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, dictionary_names) and p.requires_grad],
            "lr_scale": config["LR_DICTIONARY_SCALE"],
            "lr": config["LR"] * config["LR_DICTIONARY_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if not _match_names(n, backbone_names)
                       and not _match_names(n, linear_proj_names)
                       and not _match_names(n, dictionary_names)
                       and p.requires_grad],
        }
    ]
    return param_groups


def lr_warmup(optimizer, epoch: int, curr_iter: int, tgt_lr: float, warmup_epochs: int, num_iter_per_epoch: int):
    # min_lr = 1e-8
    total_warmup_iters = warmup_epochs * num_iter_per_epoch
    current_lr_ratio = (epoch * num_iter_per_epoch + curr_iter + 1) / total_warmup_iters
    current_lr = tgt_lr * current_lr_ratio
    for param_grop in optimizer.param_groups:
        if "lr_scale" in param_grop:
            param_grop["lr"] = current_lr * param_grop["lr_scale"]
        else:
            param_grop["lr"] = current_lr
        pass
    return


def annotations_to_flatten_detr_targets(annotations: list, device):
    """
    Args:
        annotations: annotations from the dataloader.
        device: move the targets to the device.

    Returns:
        A list of targets for the DETR model supervision, len=(B*T).
    """
    targets = []
    for annotation in annotations:      # scan by batch
        for ann in annotation:          # scan by frame
            targets.append(
                {
                    "boxes": ann["bbox"].to(device),
                    "labels": ann["category"].to(device),
                }
            )
    return targets


def build_gt_assignment(track_gt_ids, hungarian_indices, annotations_t, num_detect, device):
    """
    Build GT assignment labels for cross-entropy loss on the assignment head.

    Args:
        track_gt_ids:     list of B lists, each containing int IDs for that batch's track queries
        hungarian_indices: list of B tuples (pred_idx tensor, gt_idx tensor) from Hungarian matching
        annotations_t:    list of B annotation dicts for the current frame (has 'id' field)
        num_detect:       number of detect queries (= dustbin index)
        device:           torch device

    Returns:
        gt_assignment: [B, N_track] long tensor, values in [0, num_detect]
                       (num_detect = dustbin = no matching detect query)
    """
    B = len(track_gt_ids)
    N_track = max(len(ids) for ids in track_gt_ids) if B > 0 else 0
    gt_assignment = torch.full((B, N_track), num_detect, dtype=torch.long, device=device)

    for b in range(B):
        pred_indices, gt_indices = hungarian_indices[b]
        frame_ids = annotations_t[b]["id"].to(device)   # [N_gt] — GT IDs in this frame

        # Build map: GT identity → detect query index
        gt_id_to_detect_idx = {}
        for pred_idx, gt_idx in zip(pred_indices.tolist(), gt_indices.tolist()):
            if gt_idx < len(frame_ids):
                gt_identity = frame_ids[gt_idx].item()
                gt_id_to_detect_idx[gt_identity] = pred_idx

        for i, track_identity in enumerate(track_gt_ids[b]):
            if track_identity in gt_id_to_detect_idx:
                gt_assignment[b, i] = gt_id_to_detect_idx[track_identity]
            # else: stays as dustbin

    return gt_assignment


def teacher_forcing_update(
        model, mast_out, gt_assignment, hungarian_indices, annotations_t,
        prev_track_queries, prev_track_gt_ids, device,
):
    """
    Update track queries using GT assignment (teacher forcing).

    Updates existing tracks via TIA for matched ones; freezes dustbin-assigned tracks;
    creates new track queries via InitMLP for newly appeared targets.

    Args:
        model:              MAST model (unwrapped, has .tia and .init_mlp)
        mast_out:           output dict from forward_decoder()
        gt_assignment:      [B, N_track] or None
        hungarian_indices:  list of B tuples (pred_idx, gt_idx) from this frame's matching
        annotations_t:      list of B annotation dicts for this frame
        prev_track_queries: [B, N_track, d_model] or None
        prev_track_gt_ids:  list of B lists of ints, or None
        device:             torch device

    Returns:
        new_track_queries:  [B, N_track_new, d_model]
        new_spatial_info:   [B, N_track_new, d_model]
        new_track_gt_ids:   list of B lists of ints
    """
    from models.misc import pos_to_pos_embed

    B = mast_out["detect_outputs"].shape[0]
    d_model = mast_out["detect_outputs"].shape[-1]
    num_detect = model.num_queries

    all_tq = []
    all_si = []
    all_ids = []

    for b in range(B):
        detect_out_b = mast_out["detect_outputs"][b]   # [N_detect, d_model]
        pred_boxes_b = mast_out["pred_boxes"][b]        # [N_detect, 4]

        updated_tq = []
        updated_si = []
        updated_ids = []

        # Part 1: update existing track queries
        if prev_track_queries is not None and gt_assignment is not None:
            track_out_b = mast_out["track_outputs"][b]          # [N_track, d_model]
            assign_logits_b = mast_out["assignment_logits"][b]  # [N_track, N_detect+1]

            for i in range(prev_track_queries.shape[1]):
                gt_idx = gt_assignment[b, i].item()
                tq_i = prev_track_queries[b, i]   # [d_model]

                if gt_idx == num_detect:
                    # Dustbin: keep previous track query unchanged
                    updated_tq.append(tq_i)
                    # Reuse previous spatial info (approximate with zeros if unavailable)
                    updated_si.append(torch.zeros(d_model, device=device))
                    updated_ids.append(prev_track_gt_ids[b][i])
                else:
                    # Matched: use TIA to update
                    confidence = assign_logits_b[i, gt_idx].sigmoid().unsqueeze(0)  # [1]
                    matched_detect = detect_out_b[gt_idx].unsqueeze(0)              # [1, d_model]
                    track_q = track_out_b[i].unsqueeze(0)                           # [1, d_model]
                    updated_track = model.tia(
                        track_query=track_q,
                        matched_detect=matched_detect,
                        confidence=confidence,
                    ).squeeze(0)   # [d_model]
                    updated_tq.append(updated_track)
                    # Spatial info from matched detect bbox center (detached)
                    bbox_center = pred_boxes_b[gt_idx, :2].detach()  # [2] cx, cy
                    si = pos_to_pos_embed(
                        bbox_center.unsqueeze(0), num_pos_feats=d_model // 2
                    ).squeeze(0)   # [d_model]
                    updated_si.append(si)
                    updated_ids.append(prev_track_gt_ids[b][i])

        # Part 2: new track queries for newly appeared targets
        existing_ids = set(updated_ids)
        pred_indices, gt_indices = hungarian_indices[b]
        frame_ids = annotations_t[b]["id"].to(device)  # [N_gt]

        for pred_idx, gt_idx in zip(pred_indices.tolist(), gt_indices.tolist()):
            if gt_idx >= len(frame_ids):
                continue
            gt_identity = frame_ids[gt_idx].item()
            if gt_identity not in existing_ids:
                new_tq = model.init_mlp(
                    detect_out_b[pred_idx].unsqueeze(0)
                ).squeeze(0)   # [d_model]
                bbox_center = pred_boxes_b[pred_idx, :2].detach()
                new_si = pos_to_pos_embed(
                    bbox_center.unsqueeze(0), num_pos_feats=d_model // 2
                ).squeeze(0)
                updated_tq.append(new_tq)
                updated_si.append(new_si)
                updated_ids.append(gt_identity)
                existing_ids.add(gt_identity)

        all_tq.append(updated_tq)
        all_si.append(updated_si)
        all_ids.append(updated_ids)

    # Pad across batch
    max_tracks = max((len(tq) for tq in all_tq), default=0)
    if max_tracks == 0:
        return None, None, None

    tq_padded = torch.zeros(B, max_tracks, d_model, device=device)
    si_padded = torch.zeros(B, max_tracks, d_model, device=device)

    for b in range(B):
        n = len(all_tq[b])
        if n > 0:
            tq_padded[b, :n] = torch.stack(all_tq[b])
            si_padded[b, :n] = torch.stack(all_si[b])

    return tq_padded, si_padded, all_ids


def nested_tensor_index_select(nested_tensor: NestedTensor, dim: int, index: torch.Tensor):
    tensors, mask = nested_tensor.decompose()
    _device = tensors.device
    index = index.to(_device)
    selected_tensors = torch.index_select(input=tensors, dim=dim, index=index).contiguous()
    selected_mask = torch.index_select(input=mask, dim=dim, index=index).contiguous()
    return NestedTensor(tensors=selected_tensors, mask=selected_mask)


def tensor_dict_cat(tensor_dict1, tensor_dict2, dim=0):
    if tensor_dict1 is None or tensor_dict2 is None:
        assert tensor_dict1 is not None or tensor_dict2 is not None, "One of the tensor dict should be not None."
        return tensor_dict1 if tensor_dict2 is None else tensor_dict2
    else:
        res_tensor_dict = defaultdict()
        for k in tensor_dict1.keys():
            if isinstance(tensor_dict1[k], torch.Tensor):
                res_tensor_dict[k] = torch.cat([tensor_dict1[k], tensor_dict2[k]], dim=dim)
            elif isinstance(tensor_dict1[k], dict):
                res_tensor_dict[k] = tensor_dict_cat(tensor_dict1[k], tensor_dict2[k], dim=dim)
            elif isinstance(tensor_dict1[k], list):
                assert len(tensor_dict1[k]) == len(tensor_dict2[k]), "The list should have the same length."
                res_tensor_dict[k] = [
                    tensor_dict_cat(tensor_dict1[k][_], tensor_dict2[k][_], dim=dim)
                    for _ in range(len(tensor_dict1[k]))
                ]
            else:
                raise ValueError(f"Unsupported type {type(tensor_dict1[k])} in the tensor dict concat.")
        return dict(res_tensor_dict)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # from issue: https://github.com/pytorch/pytorch/issues/11201
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:   # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:                                   # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)

    # Call the "train_engine" function:
    train_engine(config=cfg)
