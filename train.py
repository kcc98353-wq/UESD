#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from distillation import train_with_ssd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_MAX_LENGTH = 16000 * 3



def _parse_sessions(text):
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or None


def _load_waveform(audio_path: str, max_length: int):
    audio, _sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)))
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 0.0:
        audio = audio / max_abs
    return torch.from_numpy(audio).float()


class IEMOCAPDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_length: int = DEFAULT_MAX_LENGTH,
        sessions=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = str(split)
        self.max_length = int(max_length)
        self.label_dict = {"ang": 0, "hap": 1, "neu": 2, "sad": 3}
        self.sessions = sessions
        self.audio_files, self.labels = self._load_data()
        logger.info(f"Loaded {len(self.audio_files)} samples for {self.split}")

    def _load_data(self):
        audio_files = []
        labels = []

        sessions_all = ["Session1", "Session2", "Session3", "Session4", "Session5"]
        if self.sessions is not None:
            use_sessions = list(self.sessions)
        elif self.split == "train":
            use_sessions = sessions_all[:4]
        else:
            use_sessions = [sessions_all[4]]

        for session in use_sessions:
            session_path = self.data_dir / session
            if not session_path.exists():
                continue
            eval_dir = session_path / "dialog" / "EmoEvaluation"
            if not eval_dir.exists():
                continue

            for eval_file in eval_dir.glob("*.txt"):
                if eval_file.name in ["Attribute", "Categorical", "Self-evaluation"]:
                    continue
                try:
                    with open(eval_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if not (line.startswith("[") and "]" in line and "\t" in line):
                                continue
                            parts = line.split("\t")
                            if len(parts) < 3:
                                continue
                            utterance_id = parts[1].strip()
                            emotion = parts[2].strip()
                            if emotion == "exc":
                                emotion = "hap"
                            if emotion not in self.label_dict:
                                continue
                            folder = utterance_id.rsplit("_", 1)[0]
                            wav_path = session_path / "sentences" / "wav" / folder / f"{utterance_id}.wav"
                            if wav_path.exists():
                                audio_files.append(str(wav_path))
                                labels.append(int(self.label_dict[emotion]))
                except Exception as exc:
                    logger.warning(f"Failed to read {eval_file}: {exc}")

        return audio_files, labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = int(self.labels[idx])
        try:
            waveform = _load_waveform(audio_path, self.max_length)
        except Exception as exc:
            logger.warning(f"Failed to load audio {audio_path}: {exc}")
            waveform = torch.zeros(self.max_length)
        return waveform, torch.tensor(label, dtype=torch.long)


class TeacherWithClassifier(nn.Module):
    def __init__(self, teacher_base, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.teacher_base = teacher_base
        self.pre_net = nn.Linear(768, int(hidden_dim))
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.post_net = nn.Linear(int(hidden_dim), int(num_classes))

    def forward(self, waveform, padding_mask=None):
        with torch.no_grad():
            result = self.teacher_base.extract_features(
                source=waveform,
                padding_mask=padding_mask,
                mask=False,
            )
            features = result["x"]

        x = self.activate(self.pre_net(features))
        x = self.dropout(x)
        if padding_mask is not None:
            mask = (~padding_mask).float()
            x = x * mask.unsqueeze(-1)
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.post_net(x)

    def extract_features(self, source, padding_mask=None, mask: bool = False):
        return self.teacher_base.extract_features(
            source=source,
            padding_mask=padding_mask,
            mask=mask,
        )


def load_teacher_model(model_path: str, device: str):
    try:
        import fairseq
        import fairseq.checkpoint_utils
        import fairseq.utils
        import upstream.tasks.audio_pretraining
        import upstream.models.emotion2vec

        fairseq.utils.import_user_module(
            argparse.Namespace(user_dir=os.path.join(os.path.dirname(__file__), "upstream"))
        )
        models, _cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [model_path],
            arg_overrides={"data": "/tmp"},
        )
        model = models[0]
        model.eval()
        model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        return model
    except Exception as exc:
        raise RuntimeError(f"无法加载教师模型（需要 fairseq + upstream）：{exc}")


def _resolve_hidden_dim(classifier_path: str):
    config_path = os.path.join(os.path.dirname(classifier_path), "config.json")
    if not os.path.exists(config_path):
        return 256
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg.get("hidden_dim", 256))
    except Exception:
        return 256


def load_teacher_with_classifier(base_model_path: str, classifier_path: str, device: str):
    teacher_base = load_teacher_model(base_model_path, device=device)
    hidden_dim = _resolve_hidden_dim(classifier_path)
    teacher_model = TeacherWithClassifier(teacher_base=teacher_base, hidden_dim=hidden_dim, num_classes=4).to(device)

    checkpoint = torch.load(classifier_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    candidate = {}
    for k, v in state_dict.items():
        if k.startswith("classifier."):
            candidate[k[len("classifier.") :]] = v
        else:
            candidate[k] = v

    pre_w = candidate.get("pre_net.weight")
    pre_b = candidate.get("pre_net.bias")
    post_w = candidate.get("post_net.weight")
    post_b = candidate.get("post_net.bias")

    if pre_w is None or pre_b is None or post_w is None or post_b is None:
        missing = [x for x in ["pre_net.weight", "pre_net.bias", "post_net.weight", "post_net.bias"] if candidate.get(x) is None]
        raise RuntimeError(f"分类头权重缺失: {missing}")

    teacher_model.pre_net.load_state_dict({"weight": pre_w, "bias": pre_b}, strict=True)
    teacher_model.post_net.load_state_dict({"weight": post_w, "bias": post_b}, strict=True)

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    return teacher_model


def _create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
):
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        labels = list(getattr(train_dataset, "labels", []))
        if labels:
            class_counts = Counter(labels)
            num_samples = len(labels)
            class_weights = {c: num_samples / count for c, count in class_counts.items() if count > 0}
            sample_weights = [float(class_weights.get(int(l), 0.0)) for l in labels]
            sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
            shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=True,
        )
    return train_loader, val_loader


def run_training(args):
    set_global_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    from student_model import create_student_model

    train_dataset = IEMOCAPDataset(
        data_dir=args.data_dir,
        split="train",
        max_length=DEFAULT_MAX_LENGTH,
        sessions=None,
    )
    val_dataset = IEMOCAPDataset(
        data_dir=args.data_dir,
        split="val",
        max_length=DEFAULT_MAX_LENGTH,
        sessions=None,
    )

    train_loader, val_loader = _create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    teacher_model = load_teacher_with_classifier(
        base_model_path=args.teacher_base_path,
        classifier_path=args.teacher_classifier_path,
        device=str(device),
    )

    student_model = create_student_model(
        arch="paper_distil",
        teacher_base=getattr(teacher_model, "teacher_base", None),
    ).to(device)

    train_with_ssd(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
    )

    save_path = os.path.join(args.save_dir, "student_model.pth")
    torch.save(student_model.state_dict(), save_path)
    logger.info(f"Saved student checkpoint: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--teacher_base_path", type=str, required=True)
    parser.add_argument("--teacher_classifier_path", type=str, required=True)

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
