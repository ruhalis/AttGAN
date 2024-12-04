# AttGAN
python train.py --attrs Wrinkled --data_path wrinkle_dataset/images --attr_path wrinkle_dataset/list_attr.txt --img_size 512 --shortcut_layers 1 --inject_layers 1 --mode wgan --lambda_1 100 --lambda_2 10 --lambda_3 1 --lambda_gp 10 --lr 0.0002 --epochs 200 --batch_size 32 --num_workers 4 --gpu --experiment_name experiment_001


python test_single.py --img_path 2.png --checkpoint output/experiment_001/checkpoint/weights.106.pth --img_size 512 --gpu --shortcut_layers 1 --inject_layers 1 --mode wgan --lambda_1 100 --lambda_2 10 --lambda_3 1 --lambda_gp 10



BEST 123
127


python test_single.py --img_path 2.png --checkpoint /AttGAN/output/experiment_001/checkpoint/weights.123.pth --esrgan_model /ESRGAN/ model.pth --img_size 256 --gpu

python test_single.py --img_path 2.png --checkpoint output/experiment_001/checkpoint/weights.123.pth --img_size 512 --gpu --model_name RealESRGAN_x4plus --denoise_strength 0.5 --outscale 4