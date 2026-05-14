import os
import random

import torch

from torch.utils.data import Dataset

from PIL import Image

import torchvision.transforms as T

# =========================================================
# DATASET
# =========================================================

class VideoFrameDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="Train",
        num_frames=16,
        transform=None,
        train=True
    ):

        self.root_dir = root_dir

        self.split = split

        self.num_frames = num_frames

        self.transform = transform

        self.train = train

        # -------------------------------------------------
        # CLASS LIST
        # -------------------------------------------------

        split_path = os.path.join(
            root_dir,
            split
        )

        self.classes = sorted([
            d for d in os.listdir(split_path)
            if os.path.isdir(
                os.path.join(split_path, d)
            )
        ])

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

        self.samples = []

        # =================================================
        # BUILD VIDEO GROUPS
        # =================================================

        for cls in self.classes:

            class_dir = os.path.join(
                split_path,
                cls
            )

            all_files = sorted(
                os.listdir(class_dir)
            )

            video_dict = {}

            # ---------------------------------------------
            # GROUP FRAMES INTO VIDEOS
            # ---------------------------------------------

            for frame_name in all_files:

                if not frame_name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    continue

                # -----------------------------------------
                # Example:
                # Fighting051_x264_6350.png
                #
                # becomes:
                # Fighting051_x264
                # -----------------------------------------

                stem = os.path.splitext(
                    frame_name
                )[0]

                parts = stem.split("_")

                if len(parts) < 2:
                    continue

                video_id = "_".join(parts[:-1])

                if video_id not in video_dict:
                    video_dict[video_id] = []

                video_dict[video_id].append(
                    os.path.join(
                        class_dir,
                        frame_name
                    )
                )

            # ---------------------------------------------
            # STORE VALID VIDEOS
            # ---------------------------------------------

            for video_id, frame_paths in video_dict.items():

                frame_paths = sorted(frame_paths)

                if len(frame_paths) >= num_frames:

                    self.samples.append(
                        (
                            frame_paths,
                            self.class_to_idx[cls]
                        )
                    )

        print(
            f"[{split}] Loaded {len(self.samples)} videos"
        )

    # =====================================================
    # LEN
    # =====================================================

    def __len__(self):
        return len(self.samples)

    # =====================================================
    # GETITEM
    # =====================================================

    def __getitem__(self, idx):

        frame_paths, label = self.samples[idx]

        total_frames = len(frame_paths)

        # =================================================
        # TEMPORAL SAMPLING
        # =================================================

        if self.train:

            # random clip during training
            start_idx = random.randint(
                0,
                total_frames - self.num_frames
            )

        else:

            # center clip during validation
            start_idx = (
                total_frames - self.num_frames
            ) // 2

        selected_frames = frame_paths[
            start_idx : start_idx + self.num_frames
        ]

        frames = []

        # =================================================
        # LOAD FRAMES
        # =================================================

        for frame_path in selected_frames:

            img = Image.open(
                frame_path
            ).convert("RGB")

            # ---------------------------------------------
            # LIGHT AUGMENTATION
            # ---------------------------------------------

            if self.train:

                if random.random() < 0.5:

                    img = T.functional.hflip(img)

            # ---------------------------------------------
            # RESIZE
            # ---------------------------------------------

            img = T.functional.resize(
                img,
                (112, 112)
            )

            # ---------------------------------------------
            # PIL -> Tensor
            # Shape:
            # (C, H, W)
            # ---------------------------------------------

            img = T.functional.to_tensor(img)
            img = T.functional.normalize(
                img,
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989]
                )

            frames.append(img)

        # =================================================
        # STACK FRAMES
        # Shape:
        # (T, C, H, W)
        # =================================================

        video = torch.stack(frames)

        # =================================================
        # APPLY VIDEO TRANSFORMS
        # =================================================

        if self.transform is not None:

            video = self.transform(video)

        # =================================================
        # MODEL EXPECTS:
        # (C, T, H, W)
        # =================================================

        video = video.permute(
            1,
            0,
            2,
            3
        )

        # =================================================
        # SAFETY CHECK
        # =================================================

        assert video.shape == (
            3,
            self.num_frames,
            112,
            112
        ), f"Bad shape: {video.shape}"

        return video, label

# =========================================================
# SMOKE TEST
# =========================================================

if __name__ == "__main__":

    import sys

    root = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "UCF3Class_PartialBalanced"
    )

    ds = VideoFrameDataset(
        root_dir=root,
        split="Train",
        num_frames=16,
        train=True
    )

    print("\nClasses:")
    print(ds.classes)

    print("\nNumber of samples:")
    print(len(ds))

    x, y = ds[0]

    print("\nVideo tensor shape:")
    print(x.shape)

    print("\nLabel:")
    print(y)