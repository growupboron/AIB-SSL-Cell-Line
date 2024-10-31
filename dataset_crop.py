import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up logging
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)  # Adjust as needed

    # Each process writes to its own log file
    rank = int(os.environ.get('LOCAL_RANK', 0))
    f_handler = logging.FileHandler(f"dataset_debug_rank_{rank}.log")
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

logger = setup_logging()

# Set the seed for reproducibility
seed = 42
generator = torch.Generator().manual_seed(seed)
logger.debug(f"Random seed set to {seed}")

SPLITS = {
    "train": [
        "HEPG2-01",
        "HEPG2-02",
        "HEPG2-03",
        "HEPG2-04",
        "HEPG2-05",
        "HEPG2-06",
        "HEPG2-07",
        "HUVEC-01",
        "HUVEC-02",
        "HUVEC-03",
        "HUVEC-04",
        "HUVEC-05",
        "HUVEC-06",
        "HUVEC-07",
        "HUVEC-08",
        "HUVEC-09",
        "HUVEC-10",
        "HUVEC-11",
        "HUVEC-12",
        "HUVEC-13",
        "HUVEC-14",
        "HUVEC-15",
        "HUVEC-16",
        "RPE-01",
        "RPE-02",
        "RPE-03",
        "RPE-04",
        "RPE-05",
        "RPE-06",
        "RPE-07",
        "U2OS-01",
        "U2OS-02",
        "U2OS-03",
    ],
    "val": ["HEPG2-08", "HUVEC-17", "RPE-08", "U2OS-04"],
    "test": [
        "HEPG2-09",
        "HEPG2-10",
        "HEPG2-11",
        "HUVEC-18",
        "HUVEC-19",
        "HUVEC-20",
        "HUVEC-21",
        "HUVEC-22",
        "HUVEC-23",
        "HUVEC-24",
        "RPE-09",
        "RPE-10",
        "RPE-11",
        "U2OS-05",
    ],
    "iid_val": [
        "HEPG2-01",
        "HEPG2-02",
        "HEPG2-03",
        "HEPG2-04",
        "HEPG2-05",
        "HEPG2-06",
        "HEPG2-07",
        "HUVEC-01",
        "HUVEC-02",
        "HUVEC-03",
        "HUVEC-04",
        "HUVEC-05",
        "HUVEC-06",
        "HUVEC-07",
        "HUVEC-08",
        "HUVEC-09",
        "HUVEC-10",
        "HUVEC-11",
        "HUVEC-12",
        "HUVEC-13",
        "HUVEC-14",
        "HUVEC-15",
        "HUVEC-16",
        "RPE-01",
        "RPE-02",
        "RPE-03",
        "RPE-04",
        "RPE-05",
        "RPE-06",
        "RPE-07",
        "U2OS-01",
        "U2OS-02",
        "U2OS-03",
    ],
}

SITES = {"train": [1], "val": [1, 2], "test": [1, 2], "iid_val": [2]}

class RxRx1WildsCellDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        summary_file: str,
        subset: str = "train",
        transform=None,
        num_img: int = 2,
        mode: str = "random",
        metadata_file=None,
        include_labels=False,
        cell_type=None,
        use_one_site=False,
    ):
        logger.debug(
            f"Initializing RxRx1WildsCellDataset with img_dir='{img_dir}', summary_file='{summary_file}', "
            f"subset='{subset}', transform={transform}, num_img={num_img}, mode='{mode}', "
            f"metadata_file='{metadata_file}', include_labels={include_labels}, "
            f"cell_type='{cell_type}', use_one_site={use_one_site}"
        )
        try:
            self.img_dir = img_dir
            summary_path = summary_file
            if not os.path.exists(summary_path):
                logger.error(f"Summary file not found at '{summary_path}'")
                raise FileNotFoundError(f"Summary file not found at '{summary_path}'")
            self.df = pd.read_csv(
                summary_file,
                names=["experiment", "plate", "well", "segmented", "available", "extra"],
                skiprows=1,
            )
            logger.info(f"Summary file loaded with {len(self.df)} entries")

            # Filtering dataframe
            self.df = self.df[["experiment", "plate", "well", "segmented"]]
            self.df = self.df.dropna()
            self.df = self.df[self.df["segmented"] > 0]
            logger.debug(f"Filtered dataframe to {len(self.df)} entries with 'segmented' > 0")

            self.df = self.df[self.df["experiment"].isin(SPLITS[subset])]
            logger.debug(f"Filtered dataframe for subset '{subset}' with {len(self.df)} entries")

            if cell_type is not None:
                self.df = self.df[self.df["experiment"].apply(lambda x: cell_type in x)]
                logger.debug(f"Filtered dataframe for cell_type '{cell_type}' with {len(self.df)} entries")

            if use_one_site:
                possible_sites = SITES[subset][:1]
            else:
                possible_sites = SITES[subset]
            self.df = self.df[self.df["well"].apply(lambda x: int(x[-5]) in possible_sites)]
            logger.debug(f"Filtered dataframe for sites {possible_sites} with {len(self.df)} entries")

            self.df = self.df.drop_duplicates()
            duplicates = self.df.duplicated(["experiment", "plate", "well"]).sum()
            assert duplicates == 0, f"Found {duplicates} duplicate entries after filtering"
            logger.debug("No duplicate entries found in the dataframe after filtering")

            self.df = self.df.reset_index(drop=True)
            logger.info(f"Final dataframe contains {len(self.df)} entries after all filtering and resetting index")

            self.subset = subset
            self.transform = transform
            self.num_img = num_img
            self.mode = mode
            self.metadata_file = metadata_file
            self.include_labels = include_labels

            if self.include_labels and self.metadata_file is None:
                logger.error("metadata_file must be provided if include_labels is True")
                raise ValueError("Must include metadata_file if returning labels.")

            if self.metadata_file is not None:
                metadata_path = self.metadata_file
                if not os.path.exists(metadata_path):
                    logger.error(f"Metadata file not found at '{metadata_path}'")
                    raise FileNotFoundError(f"Metadata file not found at '{metadata_path}'")
                metadata_df = pd.read_csv(metadata_file)
                logger.info(f"Metadata file loaded with {len(metadata_df)} entries")

                metadata_df["plate"] = metadata_df["plate"].apply(
                    lambda x: "Plate" + str(x)
                )
                metadata_df["well"] = metadata_df[["well", "site"]].apply(
                    lambda x: x["well"] + "_s" + str(x["site"]) + ".png", axis=1
                )
                metadata_df = metadata_df.set_index(["experiment", "plate", "well"])
                self.df = self.df.join(metadata_df, on=["experiment", "plate", "well"])
                logger.debug(f"Joined metadata with main dataframe, resulting in {len(self.df)} entries")

            self.exp_to_id = {k: i for i, k in enumerate(self.df["experiment"].unique())}
            self.plate_to_id = {k: i for i, k in enumerate(self.df["plate"].unique())}
            self.exp_plate_to_id = (
                lambda exp, plate: 4 * self.exp_to_id[exp] + self.plate_to_id[plate]
            )
            logger.debug("Mapping dictionaries for experiments and plates created")
        except Exception as e:
            logger.exception("Failed to initialize RxRx1WildsCellDataset")
            raise e

    def __len__(self):
        length = len(self.df)
        logger.debug(f"RxRx1WildsCellDataset length: {length}")
        return length

    def __getitem__(self, idx: int):
        try:
            entry = self.df.loc[idx]
            img_dir = os.path.join(
                self.img_dir, entry["experiment"], entry["plate"], entry["well"][:-4]
            )
            if not os.path.isdir(img_dir):
                logger.error(f"Image directory not found at '{img_dir}' for index {idx}")
                raise FileNotFoundError(f"Image directory not found at '{img_dir}'")

            # Seed numpy random number generator per process to avoid duplicates across processes
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            np_random_seed = idx + local_rank * 100000
            np.random.seed(np_random_seed)
            logger.debug(f"Set numpy random seed to {np_random_seed} for index {idx} on rank {local_rank}")

            if self.mode == "random":
                img_idxs = np.random.randint(entry["segmented"], size=self.num_img)
                logger.debug(f"Mode 'random': Selected image indices {img_idxs} for index {idx}")
            elif self.mode == "first":
                img_idxs = [0] * self.num_img
                logger.debug(f"Mode 'first': Selected image indices {img_idxs} for index {idx}")
            elif self.mode == "all":
                img_idxs = np.arange(int(entry["segmented"]))
                logger.debug(f"Mode 'all': Selected image indices {img_idxs} for index {idx}")
            elif self.mode == "random_single":
                chosen = np.random.randint(entry["segmented"], size=1)[0]
                img_idxs = [chosen] * self.num_img
                logger.debug(f"Mode 'random_single': Selected image index {chosen} for index {idx}")
            else:
                logger.error(f"Invalid mode '{self.mode}' specified")
                raise ValueError("Mode not implemented.")

            imgs = []
            for x in img_idxs:
                img_path = os.path.join(img_dir, f"{x}.png")
                if not os.path.exists(img_path):
                    logger.error(f"Image file not found at '{img_path}' for index {idx}")
                    raise FileNotFoundError(f"Image file not found at '{img_path}'")
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = np.array(img)  # Convert PIL Image to NumPy array
                imgs.append(img)
                logger.debug(f"Loaded and converted image '{img_path}' for index {idx}")

            if self.transform is not None:
                imgs = [self.transform(image=img)['image'] for img in imgs]
                # Convert to tensor after augmentation
                imgs = [torch.from_numpy(img).permute(2, 0, 1) for img in imgs]
                imgs = torch.stack(imgs)  # Shape: [num_img, C, H, W]
                logger.debug(f"Applied transform and stacked images for index {idx}")

            if self.include_labels:
                sirna_id = torch.tensor(entry["sirna_id"], dtype=torch.long)
                exp_plate_id = torch.tensor(self.exp_plate_to_id(entry["experiment"], entry["plate"]), dtype=torch.long)
                logger.debug(f"Included labels for index {idx}: sirna_id={sirna_id}, exp_plate_id={exp_plate_id}")
                return (
                    imgs,
                    sirna_id,
                    exp_plate_id,
                )
            else:
                exp_plate_id = torch.tensor(self.exp_plate_to_id(entry["experiment"], entry["plate"]), dtype=torch.long)
                logger.debug(f"Excluded labels for index {idx}: exp_plate_id={exp_plate_id}")
                return (
                    imgs,
                    None,
                    exp_plate_id,
                )
        except Exception as e:
            logger.exception(f"Failed to get item at index {idx}")
            raise e
