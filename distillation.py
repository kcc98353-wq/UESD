import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from torch.func import functional_call
except Exception:
    try:
        from torch.nn.utils.stateless import functional_call
    except Exception:
        try:
            from functorch import functional_call
        except Exception:
            def functional_call(*args, **kwargs):
                raise ModuleNotFoundError("functional_call is unavailable.")

from collections import OrderedDict
import numpy as np


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.temperature = float(temperature)

    def compute_kd_loss(self, student_logits, teacher_logits, temperature=None):
        if teacher_logits is None:
            return None
        t = float(temperature) if temperature is not None else self.temperature
        return (
            F.kl_div(
                F.log_softmax(student_logits / t, dim=1),
                F.softmax(teacher_logits / t, dim=1),
                reduction="batchmean",
            )
            * (t ** 2)
        )

    def forward(
        self,
        student_logits,
        teacher_logits=None,
        labels=None,
        student_features=None,
        teacher_features=None,
        student_hidden=None,
        teacher_hidden=None,
        return_components=False,
    ):
        total_loss = 0.0
        loss_details = {}

        ce_loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            total_loss = total_loss + (1.0 - self.alpha) * ce_loss
            loss_details["ce_loss"] = float(ce_loss.item())

        kd_loss = self.compute_kd_loss(student_logits, teacher_logits)
        if kd_loss is not None:
            total_loss = total_loss + self.alpha * kd_loss
            loss_details["kd_loss"] = float(kd_loss.item())

        if return_components:
            return kd_loss, None, total_loss, loss_details

        return total_loss, loss_details


class TeachingProxy(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4, dropout=0.4):
        super().__init__()
        self.pre_net = nn.Linear(input_dim, hidden_dim)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.post_net = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, padding_mask=None):
        x = self.activate(self.pre_net(features))
        x = self.dropout(x)
        if padding_mask is not None:
            mask = (~padding_mask).float()
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return self.post_net(pooled)


def _ssd_head_logits(head, features, padding_mask=None):
    x = head.activate(head.pre_net(features))
    x = head.dropout(x)
    if padding_mask is not None:
        mask = (~padding_mask).float()
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    else:
        pooled = x.mean(dim=1)
    return head.post_net(pooled)


def _ssd_split_tensor(x, support_size):
    if x is None:
        return None, None
    if isinstance(x, (list, tuple)):
        return x[:support_size], x[support_size:]
    return x[:support_size], x[support_size:]


def _ssd_kl(p_logits, q_logits, temperature):
    t = float(temperature)
    return (
        F.kl_div(
            F.log_softmax(p_logits / t, dim=1),
            F.softmax(q_logits / t, dim=1),
            reduction="batchmean",
        )
        * (t ** 2)
    )


def calculate_metrics(labels, preds, num_classes=None):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    if labels.size == 0:
        return 0.0, 0.0, 0.0

    correct = (labels == preds).astype(np.float32)
    wa = float(correct.mean())

    if num_classes is None:
        num_classes = int(max(labels.max(), preds.max()) + 1)

    ua_list = []
    f1_list = []
    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        tp = float(np.sum(preds[idx] == c))
        fn = float(np.sum(preds[idx] != c))
        fp = float(np.sum((labels != c) & (preds == c)))

        recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
        ua_list.append(recall)

        if precision + recall > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_list.append(f1)

    ua = float(np.mean(ua_list)) if ua_list else 0.0
    wf1 = float(np.mean(f1_list)) if f1_list else 0.0
    return wa, ua, wf1


def _setup_training(
    student_model,
    tp_model,
    learning_rate,
    tp_learning_rate,
    weight_decay,
    tp_weight_decay,
    alpha,
    temperature,
):
    student_optimizer = optim.AdamW(
        student_model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    tp_optimizer = optim.AdamW(
        tp_model.parameters(),
        lr=float(tp_learning_rate),
        weight_decay=float(tp_weight_decay),
    )
    distill_loss_fn = DistillationLoss(alpha=alpha, temperature=temperature)
    return student_optimizer, tp_optimizer, distill_loss_fn


def _train_epoch_ssd(
    epoch,
    loader,
    teacher_model,
    student_model,
    tp_model,
    student_optimizer,
    tp_optimizer,
    distill_loss_fn,
    noise_augmentation,
    device,
    buffers,
    support_ratio,
    inner_steps,
    inner_lr,
    anchor_weight,
    tp_grad_clip_norm,
    grad_clip_norm,
    anchor_temperature,
):
    student_model.train()
    tp_model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for clean_audio, labels in loader:
        clean_audio = clean_audio.to(device)
        labels = labels.to(device)

        if noise_augmentation is not None:
            if hasattr(noise_augmentation, "augment"):
                noisy_audio = noise_augmentation.augment(clean_audio)
            else:
                noisy_audio = noise_augmentation.add_noise(clean_audio)
        else:
            noisy_audio = clean_audio

        batch_size = noisy_audio.size(0)
        support_size = int(batch_size * float(support_ratio))
        if batch_size > 1:
            support_size = max(1, min(batch_size - 1, support_size))
        else:
            support_size = batch_size

        noisy_support, noisy_query = _ssd_split_tensor(noisy_audio, support_size)
        labels_support, labels_query = _ssd_split_tensor(labels, support_size)

        with torch.no_grad():
            teacher_outputs = teacher_model.extract_features(clean_audio, padding_mask=None, mask=False)
            teacher_features_all = teacher_outputs["x"]
            teacher_padding_all = teacher_outputs.get("padding_mask", None)

        teacher_features_support, teacher_features_query = _ssd_split_tensor(
            teacher_features_all, support_size
        )
        teacher_padding_support, teacher_padding_query = _ssd_split_tensor(
            teacher_padding_all, support_size
        )

        ta_logits_query = _ssd_head_logits(teacher_model, teacher_features_query, teacher_padding_query)
        tp_logits_support = tp_model(teacher_features_support, teacher_padding_support)
        tp_logits_query = tp_model(teacher_features_query, teacher_padding_query)

        if noisy_query is not None and noisy_query.size(0) > 0:
            fast_params = OrderedDict((name, param) for name, param in student_model.named_parameters())
            for _ in range(int(inner_steps)):
                sm_logits_sup = functional_call(
                    student_model,
                    {**buffers, **fast_params},
                    (noisy_support,),
                    {},
                )
                inner_loss, _ = distill_loss_fn(
                    student_logits=sm_logits_sup,
                    teacher_logits=tp_logits_support,
                    labels=labels_support,
                )
                trainable_items = [(n, p) for n, p in fast_params.items() if p.requires_grad]
                trainable_params = [p for _, p in trainable_items]
                if not trainable_params:
                    break
                grads = torch.autograd.grad(
                    inner_loss,
                    trainable_params,
                    create_graph=True,
                    allow_unused=True,
                )
                next_fast = OrderedDict()
                next_fast.update(fast_params)
                for (name, param), grad in zip(trainable_items, grads):
                    if grad is not None:
                        next_fast[name] = param - float(inner_lr) * grad
                fast_params = next_fast

            sm_logits_qry = functional_call(
                student_model,
                {**buffers, **fast_params},
                (noisy_query,),
                {},
            )
            val_loss = F.cross_entropy(sm_logits_qry, labels_query)
            anchor_loss = _ssd_kl(tp_logits_query, ta_logits_query, anchor_temperature)
            outer_loss = val_loss + float(anchor_weight) * anchor_loss

            tp_optimizer.zero_grad()
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(tp_model.parameters(), float(tp_grad_clip_norm))
            tp_optimizer.step()

            tp_logits_support = tp_model(teacher_features_support, teacher_padding_support)

        student_optimizer.zero_grad()
        student_logits = student_model(noisy_support)
        loss, _ = distill_loss_fn(
            student_logits=student_logits,
            teacher_logits=tp_logits_support,
            labels=labels_support,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), float(grad_clip_norm))
        student_optimizer.step()

        total_loss += float(loss.item()) * support_size
        preds = torch.argmax(student_logits, dim=1)
        total_correct += float((preds == labels_support).float().sum().item())
        total_samples += int(support_size)

    mean_loss = total_loss / float(total_samples) if total_samples > 0 else 0.0
    mean_acc = total_correct / float(total_samples) if total_samples > 0 else 0.0
    return mean_loss, mean_acc


def train_with_ssd(
    student_model,
    teacher_model,
    train_loader,
    val_loader=None,
    noise_augmentation=None,
    val_noise_augmentation=None,
    device=None,
    num_epochs=30,
    learning_rate=5e-4,
    tp_learning_rate=None,
    inner_lr=None,
    inner_steps=1,
    support_ratio=0.5,
    tp_weight_decay=1e-4,
    anchor_weight=0.1,
    anchor_temperature=None,
    weight_decay=1e-4,
    save_dir=None,
    early_stopping=False,
    early_stopping_patience=0,
    early_stopping_min_delta=0.0,
    early_stopping_metric="val_acc",
    early_stopping_mode="max",
    alpha=0.7,
    temperature=4.0,
    hidden_weight=0.0,
    student_dim=None,
    teacher_dim=768,
    feat_strategy="even_mapped_global",
    feat_num_layers=0,
    use_cfa=False,
    cfa_temperature=0.1,
    skip_snr_eval=True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    teacher_model.eval()

    hidden_dim = getattr(getattr(teacher_model, "pre_net", None), "out_features", 256)
    tp_dropout = float(getattr(getattr(teacher_model, "dropout", None), "p", 0.4))
    tp_model = TeachingProxy(
        input_dim=int(teacher_dim),
        hidden_dim=int(hidden_dim),
        num_classes=4,
        dropout=tp_dropout,
    ).to(device)

    student_optimizer, tp_optimizer, distill_loss_fn = _setup_training(
        student_model=student_model,
        tp_model=tp_model,
        learning_rate=learning_rate,
        tp_learning_rate=tp_learning_rate if tp_learning_rate is not None else learning_rate,
        weight_decay=weight_decay,
        tp_weight_decay=tp_weight_decay,
        alpha=alpha,
        temperature=temperature,
    )

    buffers = OrderedDict((name, buf) for name, buf in student_model.named_buffers())
    real_anchor_temperature = float(anchor_temperature) if anchor_temperature is not None else float(temperature)
    grad_clip_norm = 1.0
    tp_grad_clip_norm = 1.0

    for epoch in range(1, int(num_epochs) + 1):
        _train_epoch_ssd(
            epoch=epoch,
            loader=train_loader,
            teacher_model=teacher_model,
            student_model=student_model,
            tp_model=tp_model,
            student_optimizer=student_optimizer,
            tp_optimizer=tp_optimizer,
            distill_loss_fn=distill_loss_fn,
            noise_augmentation=noise_augmentation,
            device=device,
            buffers=buffers,
            support_ratio=support_ratio,
            inner_steps=inner_steps,
            inner_lr=inner_lr if inner_lr is not None else learning_rate,
            anchor_weight=anchor_weight,
            tp_grad_clip_norm=tp_grad_clip_norm,
            grad_clip_norm=grad_clip_norm,
            anchor_temperature=real_anchor_temperature,
        )

    return student_model
