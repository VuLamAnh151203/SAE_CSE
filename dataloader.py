import os
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FEATURE_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "SDT",
        "data",
        "iemocap_multimodal_features.pkl",
    )
)


def fixed_train_validation_test_split(
    train_vids,
    test_vids,
    validation_ratio=0.10,
):
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    train_vids = list(train_vids)
    test_vids = list(test_vids)
    validation_size = int(validation_ratio * len(train_vids))
    if validation_size < 1:
        raise ValueError("validation split is empty")
    if validation_size >= len(train_vids):
        raise ValueError("training split is empty")

    validation_keys = train_vids[:validation_size]
    training_keys = train_vids[validation_size:]
    testing_keys = test_vids

    training_set = set(training_keys)
    validation_set = set(validation_keys)
    testing_set = set(testing_keys)
    if len(training_set) != len(training_keys):
        raise ValueError("training dialogue IDs contain duplicates")
    if len(validation_set) != len(validation_keys):
        raise ValueError("validation dialogue IDs contain duplicates")
    if len(testing_set) != len(testing_keys):
        raise ValueError("testing dialogue IDs contain duplicates")
    if training_set & validation_set:
        raise ValueError("training and validation sets overlap")
    if training_set & testing_set:
        raise ValueError("training and testing sets overlap")
    if validation_set & testing_set:
        raise ValueError("validation and testing sets overlap")
    if training_set | validation_set != set(train_vids):
        raise ValueError("train/validation do not reconstruct trainVid")
    if testing_set != set(test_vids):
        raise ValueError("testing set does not match testVid")

    return {
        "training": training_keys,
        "validation": validation_keys,
        "testing": testing_keys,
    }


class IEMOCAPDataset(Dataset):
    """One shared dataset object containing the pickle's train and test keys."""

    def __init__(self, feature_path=DEFAULT_FEATURE_PATH):
        feature_path = os.path.abspath(feature_path)
        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                "IEMOCAP feature pickle not found: {}".format(feature_path)
            )
        with open(feature_path, "rb") as feature_file:
            values = pickle.load(feature_file, encoding="latin1")
        if len(values) != 12:
            raise ValueError(
                "expected the 12-item SDT IEMOCAP pickle schema, got {}".format(
                    len(values)
                )
            )
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.roberta2,
            self.roberta3,
            self.roberta4,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = values
        self.feature_path = feature_path
        self.keys = list(self.trainVid) + list(self.testVid)
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("trainVid and testVid overlap or contain duplicates")
        self.key_to_index = {
            key: index for index, key in enumerate(self.keys)
        }
        self._validate_labels()

    def _validate_labels(self):
        for video_id in self.keys:
            labels = self.videoLabels[video_id]
            if not labels:
                raise ValueError(
                    "dialogue {} contains no labels".format(video_id)
                )
            if min(labels) < 0 or max(labels) > 5:
                raise ValueError(
                    "dialogue {} contains a label outside [0, 5]".format(
                        video_id
                    )
                )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        video_id = self.keys[index]
        utterance_ids = self.videoIDs.get(video_id)
        if utterance_ids is None:
            utterance_ids = [
                "{}_{}".format(video_id, utterance_index)
                for utterance_index in range(
                    len(self.videoLabels[video_id])
                )
            ]
        sentences = list(self.videoSentence[video_id])
        expected_length = len(self.videoLabels[video_id])
        if len(utterance_ids) != expected_length:
            raise ValueError(
                "{} has {} utterance IDs for {} labels".format(
                    video_id, len(utterance_ids), expected_length
                )
            )
        if len(sentences) != expected_length:
            raise ValueError(
                "{} has {} sentences for {} labels".format(
                    video_id, len(sentences), expected_length
                )
            )
        return {
            "text": torch.FloatTensor(
                np.asarray(self.videoText[video_id])
            ),
            "visual": torch.FloatTensor(
                np.asarray(self.videoVisual[video_id])
            ),
            "audio": torch.FloatTensor(
                np.asarray(self.videoAudio[video_id])
            ),
            "speaker_mask": torch.FloatTensor(
                [
                    [1, 0] if speaker == "M" else [0, 1]
                    for speaker in self.videoSpeakers[video_id]
                ]
            ),
            "utterance_mask": torch.ones(
                len(self.videoLabels[video_id]), dtype=torch.float32
            ),
            "labels": torch.LongTensor(self.videoLabels[video_id]),
            "video_id": video_id,
            "utterance_ids": list(utterance_ids),
            "sentences": sentences,
        }

    @staticmethod
    def collate_fn(items):
        return {
            "text": pad_sequence(
                [item["text"] for item in items],
                batch_first=False,
            ),
            "visual": pad_sequence(
                [item["visual"] for item in items],
                batch_first=False,
            ),
            "audio": pad_sequence(
                [item["audio"] for item in items],
                batch_first=False,
            ),
            "speaker_mask": pad_sequence(
                [item["speaker_mask"] for item in items],
                batch_first=False,
            ),
            "utterance_mask": pad_sequence(
                [item["utterance_mask"] for item in items],
                batch_first=True,
            ),
            "labels": pad_sequence(
                [item["labels"] for item in items],
                batch_first=True,
            ),
            "video_ids": [item["video_id"] for item in items],
            "utterance_ids": [
                item["utterance_ids"] for item in items
            ],
            "sentences": [item["sentences"] for item in items],
        }


def create_iemocap_loaders(
    feature_path=DEFAULT_FEATURE_PATH,
    batch_size=16,
    validation_ratio=0.10,
    num_workers=0,
    pin_memory=False,
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = IEMOCAPDataset(feature_path)
    split_ids = fixed_train_validation_test_split(
        dataset.trainVid,
        dataset.testVid,
        validation_ratio=validation_ratio,
    )

    split_indices = {
        split: [dataset.key_to_index[key] for key in keys]
        for split, keys in split_ids.items()
    }
    training = DataLoader(
        Subset(dataset, split_indices["training"]),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    training_export = DataLoader(
        Subset(dataset, split_indices["training"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation = DataLoader(
        Subset(dataset, split_indices["validation"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testing = DataLoader(
        Subset(dataset, split_indices["testing"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return {
        "dataset": dataset,
        "training": training,
        "training_export": training_export,
        "validation": validation,
        "testing": testing,
        "split_ids": split_ids,
    }
