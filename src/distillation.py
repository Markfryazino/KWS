from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import wandb
from src.util_classes import count_FA_FR


def distill_epoch(student, teacher, opt, loader, teacher_log_melspec, 
                  student_log_melspec, device, distill_w=1., attn_distill_w=0,
                  log_steps=25, temperature=1., scheduler=None):
    student.to(device)
    teacher.to(device)

    kl_criterion = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')

    student.train()
    teacher.eval()

    total_cls_loss = total_kl_loss = total_attn_loss = total_loss = 0.

    for i, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch_teacher = teacher_log_melspec(batch)
        batch_student = student_log_melspec(batch)

        opt.zero_grad()

        student_logits, student_energy = student(batch_student, return_alpha=True)

        with torch.no_grad():
            teacher_logits, teacher_energy = teacher(batch_teacher, return_alpha=True)

        student_logits /= temperature
        teacher_logits /= temperature
        student_energy /= temperature
        teacher_energy /= temperature

        # we need probabilities so we use softmax & CE separately
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.softmax(teacher_logits, dim=-1)

        cls_loss = F.cross_entropy(student_logits * temperature, labels)
        # kl_loss = kl_criterion(student_log_probs, teacher_log_probs)
        kl_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction='batchmean', log_target=True)

        base_probs = F.softmax(teacher_logits, dim=1)
        kl_loss = - torch.sum(base_probs * torch.log_softmax(student_logits, dim=1)) / base_probs.size(0)
        
        loss = cls_loss + kl_loss * distill_w

        if attn_distill_w > 0:
            # student_alpha = F.log_softmax(student_energy, dim=-2)
            # teacher_alpha = F.log_softmax(teacher_energy, dim=-2)

            base_alpha = F.softmax(teacher_energy, dim=-2)
            attn_loss = - torch.sum(base_alpha * torch.log_softmax(student_energy, dim=-2)) / base_alpha.size(0)

            # attn_loss = kl_criterion(student_alpha, teacher_alpha)
            loss += attn_loss * attn_distill_w

            total_attn_loss += attn_loss.item()

        total_cls_loss += cls_loss.item()
        total_kl_loss += kl_loss.item()
        total_loss += loss.item()

        if (i + 1) % log_steps == 0 and wandb.run is not None:
            wandb.log({
                "train/cls_loss": total_cls_loss,
                "train/kl_loss": total_kl_loss,
                "train/attn_loss": total_attn_loss,
                "train/loss": total_loss,
                "train/lr": opt.param_groups[0]['lr']
            })

            total_cls_loss = total_kl_loss = total_attn_loss = total_loss = 0.

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5)

        opt.step()
        if scheduler is not None:
            scheduler.step()
