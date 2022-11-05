from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import wandb
from src.util_classes import count_FA_FR


def distill_epoch(student, teacher, opt, loader, log_melspec, device, distill_w=1., log_steps=25):
    kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")

    student.train()
    teacher.eval()

    total_cls_loss, total_kl_loss, total_loss = 0., 0., 0.

    for i, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        opt.zero_grad()

        student_logits = student(batch)

        with torch.no_grad():
            teacher_logits = teacher(batch)

        # we need probabilities so we use softmax & CE separately
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        cls_loss = F.cross_entropy(student_logits, labels)
        kl_loss = kl_criterion(student_probs, teacher_probs)
        loss = cls_loss + kl_loss * distill_w

        total_cls_loss += cls_loss.item()
        total_kl_loss += kl_loss.item()
        total_loss += loss.item()

        if (i + 1) % log_steps == 0 and wandb.run is not None:
            wandb.log({
                "train/cls_loss": total_cls_loss,
                "train/kl_loss": total_kl_loss,
                "train/loss": total_loss,
            })

            total_cls_loss, total_kl_loss, total_loss = 0., 0., 0.

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5)

        opt.step()