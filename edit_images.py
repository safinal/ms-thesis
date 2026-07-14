import os
import argparse
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from diffusers import AutoPipelineForImage2Image
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


def get_dataloader(args):
    if args.dataset == 'waterbirds':
        dataset = WaterbirdDataset(split='last_layer', transform=None, dataset_dir=args.dataset_path, num_classes=2, spuriousity=95)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=False)
    elif args.dataset == 'celeba':
        dataset = CustomCelebADataset(phase='last_layer', dataset_dir=args.dataset_path, spuriousity=95, transform=torchvision.transforms.ToTensor())
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    # elif args.dataset == 'urbancars':
    #     return get_urbancars_loaders(args.dataset_path, args.batch_size, "both")[1]

def load_editing_model(model_name):
    config = MODEL_CONFIGS.get(model_name)
    model_id = config["model_id"]
    
    # Load pipeline
    print(f"Loading {model_id}...")
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload()
    
    return pipeline, config

def tensor_to_pil(tensor):
    # Convert CHW tensor [0, 1] to PIL Image
    img = tensor.cpu().clone()
    img = img.mul(255).byte()
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)

def generate_counterfactuals(args):
    os.makedirs(os.path.join(args.output_dir, "0"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "1"), exist_ok=True)
    
    # Load model and data
    pipeline, config = load_editing_model(args.edit_model)
    dataloader = get_dataloader(args)
    edit_config = EDIT_CONFIGS[args.dataset]
    
    # Set random generator
    generator = torch.Generator(device=device).manual_seed(args.seed)
    tensor_to_pil = transforms.ToPILImage()

    print(f"Generating counterfactuals for {args.dataset} split...")
    
    for img_paths, images, labels,  in tqdm(dataloader):
        prompts = [edit_config[label] for label in labels]
        images = [tensor_to_pil(img) for img in images]
        with torch.no_grad():
            outputs = pipeline(
                prompt=prompts,
                image=images,
                num_inference_steps=config["num_inference_steps"],
                guidance_scale=config["guidance_scale"],
                generator=generator
            ).images
        
        for i, edited_img in enumerate(outputs):
            img_path = img_paths[i]
            img_name = os.path.basename(img_path)
            shutil.copyfile(img_path, os.path.join(args.output_dir, f"{labels[i]}", img_name))
            img_name, img_extention = os.path.splitext(img_name)
            save_path = os.path.join(args.output_dir, f"{labels[i]}", f"{img_name}_aug{img_extention}")
            edited_img.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate counterfactual images")
    parser.add_argument("--dataset", type=str, required=True, choices=["celeba", "waterbirds", "urbancars"])
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--edit_model", type=str, default="flux2-klein-4b")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    generate_counterfactuals(args)
