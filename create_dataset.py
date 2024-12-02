import os
import shutil
from PIL import Image

# Define paths here
ORIGINAL_DIR = "D:/AttGAN/original"  # Directory with wrinkled images
AFTER_DIR = "D:/AttGAN/after"        # Directory with non-wrinkled images
OUTPUT_DIR = "D:/AttGAN/wrinkle_dataset"  # Output directory

def get_file_name_without_extension(filename):
    """Get filename without extension."""
    return os.path.splitext(filename)[0]

def create_wrinkle_dataset(original_dir, after_dir, output_dir):
    """
    Create a dataset for wrinkle removal training.
    
    Args:
        original_dir: Directory containing original (wrinkled) images
        after_dir: Directory containing after (non-wrinkled) images
        output_dir: Output directory for the dataset
    """
    # Verify input directories exist
    if not os.path.exists(original_dir):
        raise FileNotFoundError(f"Original directory not found: {original_dir}")
    if not os.path.exists(after_dir):
        raise FileNotFoundError(f"After directory not found: {after_dir}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Get list of images
    original_images = sorted([f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    after_images = sorted([f for f in os.listdir(after_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Create dictionaries of filenames without extensions
    original_dict = {get_file_name_without_extension(f): f for f in original_images}
    after_dict = {get_file_name_without_extension(f): f for f in after_images}
    
    # Find common images (paired images)
    common_names = set(original_dict.keys()) & set(after_dict.keys())
    
    if not common_names:
        raise ValueError("No paired images found! Make sure the filenames match between original and after directories.")
    
    # Print unpaired images for information
    unpaired_original = set(original_dict.keys()) - common_names
    unpaired_after = set(after_dict.keys()) - common_names
    
    if unpaired_original:
        print("\nUnpaired images in original directory (these will be skipped):")
        for name in sorted(unpaired_original):
            print(f"- {original_dict[name]}")
    
    if unpaired_after:
        print("\nUnpaired images in after directory (these will be skipped):")
        for name in sorted(unpaired_after):
            print(f"- {after_dict[name]}")
    
    # Create attribute file content
    attr_lines = []
    attr_lines.append(str(len(common_names) * 2))  # Total number of images (paired)
    attr_lines.append('Wrinkled')  # Attribute name
    
    # Process and copy paired images
    for idx, base_name in enumerate(sorted(common_names)):
        # Process original image
        orig_src_path = os.path.join(original_dir, original_dict[base_name])
        orig_new_name = f'original_{idx:04d}.jpg'
        orig_dst_path = os.path.join(images_dir, orig_new_name)
        
        # Convert original image to RGB and save as JPEG
        img = Image.open(orig_src_path).convert('RGB')
        img.save(orig_dst_path, 'JPEG')
        attr_lines.append(f'{orig_new_name} 1')
        
        # Process after image
        after_src_path = os.path.join(after_dir, after_dict[base_name])
        after_new_name = f'after_{idx:04d}.jpg'
        after_dst_path = os.path.join(images_dir, after_new_name)
        
        # Convert after image to RGB and save as JPEG
        img = Image.open(after_src_path).convert('RGB')
        img.save(after_dst_path, 'JPEG')
        attr_lines.append(f'{after_new_name} -1')
    
    # Write attribute file
    attr_path = os.path.join(output_dir, 'list_attr.txt')
    with open(attr_path, 'w') as f:
        f.write('\n'.join(attr_lines))
    
    print(f"\nDataset created at {output_dir}")
    print(f"Total paired images: {len(common_names)}")
    print(f"Total images in dataset: {len(common_names) * 2}")
    if unpaired_original or unpaired_after:
        print(f"Skipped unpaired images: {len(unpaired_original)} from original, {len(unpaired_after)} from after")

if __name__ == '__main__':
    create_wrinkle_dataset(ORIGINAL_DIR, AFTER_DIR, OUTPUT_DIR)