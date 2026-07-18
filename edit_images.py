import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoPipelineForImage2Image, Flux2KleinPipeline
import shutil
from data import CelebADataset, WaterbirdDataset
import types
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomCelebADataset(CelebADataset):
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.y_array[idx]
        return img_path, img, label

EDIT_CONFIGS = {
    "celeba": {
        0: "Change the hair color of this person to Blond, keep everything else the in the photo the same, only change the hair color.",
        1: "Change the hair color of this person to a Non-Blond color (Black/Brown/Gray), keep everything else the in the photo the same, only change the hair color."
    },
    # "waterbirds": {
    #     0: "",
    #     1: ""
    # },
    # "urbancars": {
    #     0: "Replace the vehicle in this image with a sleek urban or city car (like a compact sedan or sports car). The background buildings, streets, nature, and any co-occurring objects must remain absolutely identical.",
    #     1: "Replace the vehicle in this image with a rugged country or rural vehicle (like a pickup truck or off-road SUV). The background buildings, streets, nature, and any co-occurring objects must remain absolutely identical."
    # }
}

MODEL_CONFIGS = {
    "flux2-klein-4b": {
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
    }
}


def pil_collate_fn(batch):
    """
    Tells the DataLoader how to batch our data without trying to
    convert PIL images into PyTorch tensors.
    """
    img_paths = [item[0] for item in batch]
    images = [item[1] for item in batch] # List of PIL images
    labels = torch.tensor([item[2] for item in batch])
    return img_paths, images, labels

def get_dataloader(args):
    if args.dataset == 'celeba':
        dataset = CustomCelebADataset(phase='last_layer', dataset_dir=args.dataset_path, spuriousity=95, transform=None)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=pil_collate_fn)
        return dataloader
    # elif args.dataset == 'waterbirds':
    #     dataset = WaterbirdDataset(split='last_layer', transform=torchvision.transforms.ToTensor(), dataset_dir=args.dataset_path, num_classes=2, spuriousity=95)
    #     dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    #     return dataloader, dataset
    # elif args.dataset == 'urbancars':
    #     return get_urbancars_loaders(args.dataset_path, args.batch_size, "both")[1]


def _prepare_image_latents_independent(self, images, batch_size, generator, device, dtype):
    """
    Patched version of prepare_image_latents for independent per-image editing.

    The original pools ALL images into one shared reference and repeats it
    identically for every batch element. This version instead gives each
    batch element ONLY its own image's latent as the reference context,
    enabling true GPU-batched independent edits.

    Requires len(images) == batch_size (one reference image per prompt).
    """
    assert len(images) == batch_size, (
        f"Independent batching requires len(images) == batch_size, "
        f"got {len(images)} images and batch_size={batch_size}"
    )

    per_image_packed = []
    per_image_ids = []

    for image in images:
        image = image.to(device=device, dtype=dtype)
        latent = self._encode_vae_image(image=image, generator=generator)  # (1, C, H, W)
        packed = self._pack_latents(latent)  # (1, seq_len, C)

        # Prepare positional IDs for this single image
        ids = self._prepare_image_ids([latent])  # (1, seq_len, 4)

        per_image_packed.append(packed)       # (1, seq_len, C)
        per_image_ids.append(ids)             # (1, seq_len, 4)

    # Stack along batch dim — each batch element has its own reference
    image_latents = torch.cat(per_image_packed, dim=0)  # (B, seq_len, C)
    image_latent_ids = torch.cat(per_image_ids, dim=0)  # (B, seq_len, 4)
    image_latent_ids = image_latent_ids.to(device)

    return image_latents, image_latent_ids


def load_editing_model(model_name):
    config = MODEL_CONFIGS.get(model_name)
    model_id = config["model_id"]
    
    # Load pipeline
    print(f"Loading {model_id}...")
    pipeline = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    try:
        pipeline.to(device)
    except:
        pipeline.enable_model_cpu_offload()

    # Monkey-patch to enable batched independent image editing
    pipeline.prepare_image_latents = types.MethodType(
        _prepare_image_latents_independent, pipeline
    )
    
    return pipeline, config

def generate_counterfactuals(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    pipeline, config = load_editing_model(args.edit_model)
    dataloader = get_dataloader(args)
    edit_config = EDIT_CONFIGS[args.dataset]

    print(f"Generating counterfactuals for {args.dataset} split...")
    
    generator = torch.Generator(device=device).manual_seed(args.seed)

    metadata_df = pd.DataFrame(columns=['img_filename', 'y'])

    for img_paths, images, labels in tqdm(dataloader):
        prompts = [edit_config[label.item()] for label in labels]
        
        outputs = pipeline(
            prompt=prompts,
            image=images,
            num_inference_steps=config["num_inference_steps"],
            guidance_scale=config["guidance_scale"],
            generator=generator,
        ).images

        img_name_lst = []
        labels_lst = []
        for edited_img, img_path, label in zip(outputs, img_paths, labels):
            img_name = os.path.basename(img_path)
            shutil.copyfile(img_path, os.path.join(args.output_dir, img_name))
            img_name_lst.append(img_name)
            labels_lst.append(label.item())
            img_name, img_extention = os.path.splitext(img_name)
            img_name = f"{img_name}_aug{img_extention}"
            edited_img.save(os.path.join(args.output_dir, img_name))
            img_name_lst.append(img_name)
            labels_lst.append(int(not label.item()))
        metadata_df = pd.concat((metadata_df, pd.DataFrame({'img_filename': img_name_lst, 'y': labels_lst}))).sample(frac=1)
        metadata_df.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate counterfactual images")
    parser.add_argument("--dataset", type=str, required=True, choices=["celeba", "waterbirds", "urbancars"])
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--edit_model", type=str, default="flux2-klein-4b")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    generate_counterfactuals(args)
