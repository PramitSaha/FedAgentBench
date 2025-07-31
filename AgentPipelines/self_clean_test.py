import os
import json
import shutil
import torch
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from selfclean import SelfClean
from selfclean.cleaner.selfclean import PretrainingType, DINO_STANDARD_HYPERPARAMETERS
import copy#
import selfclean
print(selfclean.__file__)
import ssl
from selfclean.core.src.pkg.embedder import Embedder
from types import SimpleNamespace
import numpy

# Allow numpy scalar unpickling
torch.serialization.add_safe_globals({
    'numpy.core.multiarray.scalar': numpy.core.multiarray.scalar
})

# ========== Paths ==========
clean_output_dir = "..."
dataset_root = "..."

# ======= Load dataset directly =======
print("🔄 Loading dataset with ImageFolder...")
dataset = ImageFolder(root=dataset_root)


# ========== Step 4: Run SelfClean ==========
parameters = copy.deepcopy(DINO_STANDARD_HYPERPARAMETERS)
parameters['model']['base_model'] = 'pretrained_imagenet_vit_tiny'

print("🚀 Running SelfClean...")
selfclean = SelfClean(
    plot_top_N=20,
    auto_cleaning=True,
)
print("Selfclean loaded")

def patched_load_pretrained(model_name=None, work_dir=None, **kwargs):
    print("🔁 Using locally downloaded DINO checkpoint")
    local_model_path = "..."

    model = Embedder.load_dino(ckp_path=local_model_path)
    dummy_config = SimpleNamespace(model_type="ViT")
    dummy_augment_fn = lambda x: x

    return model, dummy_config, dummy_augment_fn


Embedder.load_pretrained = patched_load_pretrained

issues = selfclean.run_on_dataset(
    dataset=copy.copy(dataset),
    pretraining_type=PretrainingType.DINO,
    epochs=10,
    batch_size=16,
    save_every_n_epochs=1,
    dataset_name="...",
    work_dir=clean_output_dir,
)

# print(f"✅ SelfClean finished. Issues found: {len(issues)}")
df_near_duplicates = issues.get_issues("near_duplicates", return_as_df=True)
df_off_topic_samples = issues.get_issues("off_topic_samples", return_as_df=True)
df_label_errors = issues.get_issues("label_errors", return_as_df=True)

# # Save near duplicates
df_near_duplicates.to_csv(".../near_duplicates.csv", index=False)

# Save off-topic samples
df_off_topic_samples.to_csv(".../off_topic_samples.csv", index=False)

# Save label errors
df_label_errors.to_csv(".../label_errors.csv", index=False)