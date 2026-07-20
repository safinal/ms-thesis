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
import random


device = "cuda" if torch.cuda.is_available() else "cpu"

WATERBIRD_SPECIES = [
    'Arctic Tern',
    'Black Tern',
    'Black footed Albatross',
    'Brandt Cormorant',
    'Brown Pelican',
    'California Gull',
    'Caspian Tern',
    'Common Tern',
    'Crested Auklet',
    'Eared Grebe',
    'Elegant Tern',
    'Forsters Tern',
    'Frigatebird',
    'Gadwall',
    'Glaucous winged Gull',
    'Heermann Gull',
    'Herring Gull',
    'Hooded Merganser',
    'Horned Grebe',
    'Horned Puffin',
    'Ivory Gull',
    'Laysan Albatross',
    'Least Auklet',
    'Least Tern',
    'Long tailed Jaeger',
    'Mallard',
    'Northern Fulmar',
    'Pacific Loon',
    'Parakeet Auklet',
    'Pelagic Cormorant',
    'Pied billed Grebe',
    'Pigeon Guillemot',
    'Pomarine Jaeger',
    'Red breasted Merganser',
    'Red faced Cormorant',
    'Red legged Kittiwake',
    'Rhinoceros Auklet',
    'Ring billed Gull',
    'Slaty backed Gull',
    'Sooty Albatross',
    'Western Grebe',
    'Western Gull',
    'White Pelican',
    #  'Western Meadowlark',
    #  'Western Wood Pewee',
    #  'Eastern Towhee',
]

LANDBIRD_SPECIES = [
    'Acadian Flycatcher',
    'American Crow',
    'American Goldfinch',
    'American Pipit',
    'American Redstart',
    'American Three toed Woodpecker',
    'Anna Hummingbird',
    'Baird Sparrow',
    'Baltimore Oriole',
    'Bank Swallow',
    'Barn Swallow',
    'Bay breasted Warbler',
    'Bewick Wren',
    'Black and white Warbler',
    'Black billed Cuckoo',
    'Black capped Vireo',
    'Black throated Blue Warbler',
    'Black throated Sparrow',
    'Blue Grosbeak',
    'Blue Jay',
    'Blue headed Vireo',
    'Blue winged Warbler',
    'Boat tailed Grackle',
    'Bobolink',
    'Bohemian Waxwing',
    'Brewer Blackbird',
    'Brewer Sparrow',
    'Bronzed Cowbird',
    'Brown Creeper',
    'Brown Thrasher',
    'Cactus Wren',
    'Canada Warbler',
    'Cape Glossy Starling',
    'Cape May Warbler',
    'Cardinal',
    'Carolina Wren',
    'Cedar Waxwing',
    'Cerulean Warbler',
    'Chestnut sided Warbler',
    'Chipping Sparrow',
    'Chuck will Widow',
    'Clark Nutcracker',
    'Clay colored Sparrow',
    'Cliff Swallow',
    'Common Raven',
    'Common Yellowthroat',
    'Dark eyed Junco',
    'Downy Woodpecker',
    'European Goldfinch',
    'Evening Grosbeak',
    'Field Sparrow',
    'Fish Crow',
    'Florida Jay',
    'Fox Sparrow',
    'Geococcyx',
    'Golden winged Warbler',
    'Grasshopper Sparrow',
    'Gray Catbird',
    'Gray Kingbird',
    'Gray crowned Rosy Finch',
    'Great Crested Flycatcher',
    'Great Grey Shrike',
    'Green Jay',
    'Green Violetear',
    'Green tailed Towhee',
    'Groove billed Ani',
    'Harris Sparrow',
    'Henslow Sparrow',
    'Hooded Oriole',
    'Hooded Warbler',
    'Horned Lark',
    'House Sparrow',
    'House Wren',
    'Indigo Bunting',
    'Kentucky Warbler',
    'Lazuli Bunting',
    'Le Conte Sparrow',
    'Least Flycatcher',
    'Lincoln Sparrow',
    'Loggerhead Shrike',
    'Louisiana Waterthrush',
    'Magnolia Warbler',
    'Mangrove Cuckoo',
    'Marsh Wren',
    'Mockingbird',
    'Mourning Warbler',
    'Myrtle Warbler',
    'Nashville Warbler',
    'Nelson Sharp tailed Sparrow',
    'Nighthawk',
    'Northern Flicker',
    'Northern Waterthrush',
    'Olive sided Flycatcher',
    'Orange crowned Warbler',
    'Orchard Oriole',
    'Ovenbird',
    'Painted Bunting',
    'Palm Warbler',
    'Philadelphia Vireo',
    'Pileated Woodpecker',
    'Pine Grosbeak',
    'Pine Warbler',
    'Prairie Warbler',
    'Prothonotary Warbler',
    'Purple Finch',
    'Red bellied Woodpecker',
    'Red cockaded Woodpecker',
    'Red eyed Vireo',
    'Red headed Woodpecker',
    'Red winged Blackbird',
    'Rock Wren',
    'Rose breasted Grosbeak',
    'Ruby throated Hummingbird',
    'Rufous Hummingbird',
    'Rusty Blackbird',
    'Sage Thrasher',
    'Savannah Sparrow',
    'Sayornis',
    'Scarlet Tanager',
    'Scissor tailed Flycatcher',
    'Scott Oriole',
    'Seaside Sparrow',
    'Shiny Cowbird',
    'Song Sparrow',
    'Spotted Catbird',
    'Summer Tanager',
    'Swainson Warbler',
    'Tennessee Warbler',
    'Tree Sparrow',
    'Tree Swallow',
    'Tropical Kingbird',
    'Vermilion Flycatcher',
    'Vesper Sparrow',
    'Warbling Vireo',
    'Whip poor Will',
    'White breasted Nuthatch',
    'White crowned Sparrow',
    'White eyed Vireo',
    'White necked Raven',
    'White throated Sparrow',
    'Wilson Warbler',
    'Winter Wren',
    'Worm eating Warbler',
    'Yellow Warbler',
    'Yellow bellied Flycatcher',
    'Yellow billed Cuckoo',
    'Yellow breasted Chat',
    'Yellow headed Blackbird',
    'Yellow throated Vireo',
    # 'Belted Kingfisher',
    # 'Ringed Kingfisher',
    # 'Green Kingfisher',
    # 'Pied Kingfisher',
    # 'White breasted Kingfisher',
]

def build_waterbird_prompt(source_species, target_species):
    return (
        f"This photo contains a {source_species}. "
        f"Replace the {source_species} with a {target_species}. "
        f"The new bird must clearly be a {target_species}. "
        f"Keep the background, lighting, and composition exactly the same. "
        f"Do not alter the environment or any other element in the image."
    )

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
    },
    "flux2-klein-9b": {
        "model_id": "black-forest-labs/FLUX.2-klein-9B",
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
    }
}

class CustomCelebADataset(CelebADataset):
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        label = self.y_array[idx]
        return img_path, img, label


class CustomWaterbirdDataset(WaterbirdDataset):
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB').resize((256, 256))
        label = self.y_array[idx]
        species = self.filename_array[idx].split('/')[0].split('.')[1].replace('_', ' ')
        return img_path, img, label, species

def waterbird_collate_fn(batch):
    """Collate that keeps PIL images and species strings intact."""
    img_paths = [item[0] for item in batch]
    images    = [item[1] for item in batch]
    labels    = torch.tensor([item[2] for item in batch])
    species   = [item[3] for item in batch]
    return img_paths, images, labels, species

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
    elif args.dataset == 'waterbirds':
        dataset = CustomWaterbirdDataset(split='last_layer', transform=None, dataset_dir=args.dataset_path, num_classes=2, spuriousity=95)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=waterbird_collate_fn)
        return dataloader
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
    edit_config = EDIT_CONFIGS.get(args.dataset)

    print(f"Generating counterfactuals for {args.dataset} split...")
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    species_rng = random.Random(args.seed)
    metadata_df = pd.DataFrame(columns=['img_filename', 'y', 'species'])

    for data in tqdm(dataloader):
        if args.dataset == 'waterbirds':
            img_paths, images, labels, species_names = data
            prompts = []
            species_lst = []
            for label, src_species in zip(labels, species_names):
                source_is_waterbird = (label.item() == 1)
                tgt_species = species_rng.choice(LANDBIRD_SPECIES if source_is_waterbird else WATERBIRD_SPECIES)
                species_lst += [src_species, tgt_species]
                prompts.append(build_waterbird_prompt(src_species, tgt_species))
        else:
            img_paths, images, labels = data
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

        metadata_df = pd.concat(
            (
                metadata_df, 
                pd.DataFrame({'img_filename': img_name_lst, 'y': labels_lst, 'species': species_lst})
            )
        ).sample(frac=1)
        metadata_df.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate counterfactual images")
    parser.add_argument("--dataset", type=str, required=True, choices=["celeba", "waterbirds", "urbancars"])
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--edit_model", type=str, required=True, choices=["flux2-klein-9b", "flux2-klein-4b"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    generate_counterfactuals(args)
