import os
import argparse
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from diffusers import AutoPipelineForImage2Image, Flux2KleinPipeline
import shutil
import math
from data import celebADataset, WaterbirdDataset


device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomCelebADataset(celebADataset):
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
        1: "Change the hair color of this person to Non-Blond, keep everything else the in the photo the same, only change the hair color."
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
        return dataloader, dataset
    # elif args.dataset == 'waterbirds':
    #     dataset = WaterbirdDataset(split='last_layer', transform=torchvision.transforms.ToTensor(), dataset_dir=args.dataset_path, num_classes=2, spuriousity=95)
    #     dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    #     return dataloader, dataset
    # elif args.dataset == 'urbancars':
    #     return get_urbancars_loaders(args.dataset_path, args.batch_size, "both")[1]

def load_editing_model(model_name):
    config = MODEL_CONFIGS.get(model_name)
    model_id = config["model_id"]
    
    # Load pipeline
    print(f"Loading {model_id}...")
    pipeline = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload()
    
    return pipeline, config

def generate_counterfactuals(args):
    os.makedirs(os.path.join(args.output_dir, "0"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "1"), exist_ok=True)
    
    # Load model and data
    pipeline, config = load_editing_model(args.edit_model)
    dataloader, dataset = get_dataloader(args)
    edit_config = EDIT_CONFIGS[args.dataset]

    print(f"Generating counterfactuals for {args.dataset} split...")
    
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for img_paths, images, labels in tqdm(dataloader):
        prompts = [edit_config[label.item()] for label in labels]
        
        with torch.no_grad():
            outputs = pipeline(
                prompt=prompts,
                image=images,
                num_inference_steps=config["num_inference_steps"],
                guidance_scale=config["guidance_scale"],
                generator=generator
            ).images

        for edited_img, img_path, label in zip(outputs, img_paths, labels):
            img_name = os.path.basename(img_path)
            shutil.copyfile(img_path, os.path.join(args.output_dir, f"{label.item()}", img_name))
            img_name, img_extention = os.path.splitext(img_name)
            save_path = os.path.join(args.output_dir, f"{label.item()}", f"{img_name}_aug{img_extention}")
            edited_img.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate counterfactual images")
    parser.add_argument("--dataset", type=str, required=True, choices=["celeba", "waterbirds", "urbancars"])
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--edit_model", type=str, default="flux2-klein-4b")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    generate_counterfactuals(args)
