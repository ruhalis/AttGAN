import torch
from torchvision import transforms
from PIL import Image
import os
from attgan import AttGAN
import argparse
import cv2
import numpy as np
import sys
sys.path.append('Real-ESRGAN')
from inference_realesrgan_from_RESRGAN import RealESRGAN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_path', type=str, default='output/results', help='Path to save results')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus', 
                       help='Model names: RealESRGAN_x4plus | RealESRGAN_x4plus_anime_6B')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--attrs', type=str, nargs='+', default=['Wrinkled'])
    parser.add_argument('--n_attrs', type=int, default=1)
    parser.add_argument('--mode', type=str, default='wgan')
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
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for Adam optimizer')
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
    args.betas = (args.beta1, args.beta2)
    os.makedirs(args.output_path, exist_ok=True)
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load AttGAN model
    print("Loading AttGAN model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = AttGAN(args)
    model.G.load_state_dict(checkpoint['G'])
    model.G.eval().to(device)
    print("AttGAN model loaded successfully!")
    
    # Load Real-ESRGAN model
    print("Loading Real-ESRGAN model...")
    model_path = f'weights/{args.model_name}.pth'
    upsampler = RealESRGAN(device, scale=4)
    upsampler.load_weights(model_path, download=True)
    print("Real-ESRGAN model loaded successfully!")
    
    # Process image
    print("Processing image...")
    img = load_image(args.img_path, args.img_size).to(device)
    with torch.no_grad():
        att_b = torch.tensor([[1]]).float().to(device)  # Example attribute
        att_b_ = (att_b * 2 - 1) * 0.5
        img_fake = model.G(img, att_b_)
    
    # Save AttGAN result
    base_name = os.path.splitext(os.path.basename(args.img_path))[0]
    attgan_output_path = os.path.join(args.output_path, f'{base_name}_transformed.png')
    save_image(img_fake, attgan_output_path)
    
    # Post-process with Real-ESRGAN
    print("Applying Real-ESRGAN upscaling...")
    # Convert tensor to numpy array for Real-ESRGAN
    img_np = img_fake.cpu().numpy().squeeze(0).transpose(1, 2, 0)
    img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Process with Real-ESRGAN
    sr_image = upsampler.predict(img_np)
    
    # Save Real-ESRGAN result
    realesrgan_output_path = os.path.join(args.output_path, f'{base_name}_realesrgan.png')
    cv2.imwrite(realesrgan_output_path, cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR))
    
    print(f"Results saved in {args.output_path}")
    print(f"1. AttGAN output: {attgan_output_path}")
    print(f"2. Real-ESRGAN output: {realesrgan_output_path}")

if __name__ == '__main__':
    main() 