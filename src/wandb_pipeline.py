from src.util_classes import validation, train_epoch
from src.evaluation import memory_size, n_parameters, get_mean_macs, compute_time
from src.model import CRNN
from src.distillation import distill_epoch

from tqdm.auto import tqdm
from pprint import pprint

import torch
import numpy as np
import random
import wandb
import dataclasses


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate_model(model, loader, log_melspec, device, random_state=42):
    set_random_seed(random_state)
    model.eval()

    metric = validation(model, loader, log_melspec, device)
    final_metrics = {
        "area under FA/FR curve": metric,
        "evaluation time (s)": compute_time(model, loader, log_melspec, device),
        "memory size (MB)": memory_size(model),
        "number of parameters": n_parameters(model),
        "MACs": get_mean_macs(model, loader, log_melspec, device),
    }

    if wandb.run is not None:
        wandb.log({f"test: {k}": v for k, v in final_metrics.items()})
    
    return final_metrics


def train_baseline(train_loader, val_loader, melspec_train, melspec_val, config, log_wandb=True, name_wandb=None):
    if log_wandb:
        wandb.init(
            project="kws",
            entity="broccoliman",
            name=name_wandb,
            config=dataclasses.asdict(config),
        )

    set_random_seed(config.random_state)
    model = CRNN(config).to(config.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    print("Number of trainable parameters:", n_parameters(model))

    for i in range(config.num_epochs):
        print(f"EPOCH: {i}")
        train_epoch(model, optimizer, train_loader, melspec_train, config.device)
        metric = validation(model, val_loader, melspec_val, config.device)

        if not log_wandb:
            print(metric)
        else:
            wandb.log({"val_metric": metric}, step=i)

    model_name = name_wandb if name_wandb is not None else "model"
    torch.save(model.state_dict(), f"{model_name}.pt")

    final_metrics = evaluate_model(model, val_loader, melspec_val, config.device)
    if not log_wandb:
        pprint(final_metrics)
    else:
        wandb.save(f"{model_name}.pt")
        wandb.finish()

    return model


def train_distillation(teacher, train_loader, val_loader, teacher_melspec_train,
                       student_melspec_train, student_melspec_val, config, log_wandb=True, name_wandb=None,
                       student=None):
    if log_wandb:
        wandb.init(
            project="kws",
            entity="broccoliman",
            name=name_wandb,
            config=dataclasses.asdict(config),
        )

    set_random_seed(config.random_state)

    model = CRNN(config).to(config.device) if student is None else student

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    print("Number of trainable parameters:", n_parameters(model))

    student_melspec_val.melspec.to(config.device)
    for i in range(config.num_epochs):
        print(f"EPOCH: {i}")
        distill_epoch(model, teacher, optimizer, train_loader, teacher_melspec_train, student_melspec_train,
                      config.device, config.distill_w, config.attn_distill_w)
        metric = validation(model, val_loader, student_melspec_val, config.device)

        if not log_wandb:
            print(metric)
        else:
            wandb.log({
                "val_metric": metric,
                "epoch": i
            })

    model_name = name_wandb if name_wandb is not None else "model"
    torch.save(model.state_dict(), f"{model_name}.pt")

    student_melspec_val.melspec.cpu()
    final_metrics = evaluate_model(model.cpu(), val_loader, student_melspec_val, "cpu")
    if not log_wandb:
        pprint(final_metrics)
    else:
        wandb.save(f"{model_name}.pt")
        wandb.finish()

    return model
