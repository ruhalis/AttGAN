import torch
from torchvision import transforms
from PIL import Image
import os
from attgan import AttGAN
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_path', type=str, default='output/results', help='Path to save results')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    
    # Model architecture arguments
    parser.add_argument('--attrs', type=str, nargs='+', default=['Wrinkled'])
    parser.add_argument('--n_attrs', type=int, default=1)
    parser.add_argument('--enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', type=str, default='relu')
    parser.add_argument('--shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='wgan')
    
    # Add lambda parameters
    parser.add_argument('--lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', type=float, default=10.0)
    
    # Add optimizer parameters
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=0.0002)
    
    return parser.parse_args()

def load_image(img_path, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    return img

def save_image(tensor, path):
    transform = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # [-1,1] -> [0,1]
        transforms.ToPILImage()
    ])
    
    img = tensor.cpu().squeeze(0)
    img = transform(img)
    img.save(path)

def main():
    args = parse_args()
    
    # Add betas tuple
    args.betas = (args.beta1, args.beta2)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model with command line arguments
    model = AttGAN(args)
    
    # Load only the model weights
    if 'G' in checkpoint:
        model.G.load_state_dict(checkpoint['G'])
    elif 'state_dict' in checkpoint:
        model.G.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint is just the state dict
        model.G.load_state_dict(checkpoint)
    
    model.G.eval()
    model.G.to(device)
    print("Model loaded successfully!")
    
    # Load and preprocess image
    print("Processing image...")
    img = load_image(args.img_path, args.img_size)
    img = img.to(device)
    
    # Generate attribute vectors
    att_a = torch.tensor([[0]]).float().to(device)  # Original attribute
    att_b = torch.tensor([[1]]).float().to(device)  # Target attribute
    
    # Generate transformed image
    with torch.no_grad():
        att_b_ = (att_b * 2 - 1) * 0.5
        img_fake = model.G(img, att_b_)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(args.img_path))[0]
    save_image(img, os.path.join(args.output_path, f'{base_name}_original.png'))
    save_image(img_fake, os.path.join(args.output_path, f'{base_name}_transformed.png'))
    print(f"Results saved in {args.output_path}")

if __name__ == '__main__':
    main() 