import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
torch.backends.cudnn.benchmark = True
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from dataset_load import Dataload
import time
import utils
from model import model
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from losses import *
from torch.utils.tensorboard import SummaryWriter
import argparse
from spikingjelly.activation_based import functional
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils

def save_sample_images(input_img, target_img, restored_img, save_path, epoch, batch_idx, num_samples=4):
    """
    Save sample images (input, target, restored) for visualization
    Stitch multiple samples into one row for easy comparison
    """
    # Select first num_samples from the batch
    num_samples = min(num_samples, input_img.shape[0])
    
    # Prepare images for stitching
    input_samples = input_img[:num_samples]
    target_samples = target_img[:num_samples]
    restored_samples = restored_img[:num_samples]
    
    # Create a grid with 3 rows (input, target, restored) and num_samples columns
    grid_images = []
    
    # Input images row
    grid_images.append(input_samples)
    # Target images row  
    grid_images.append(target_samples)
    # Restored images row
    grid_images.append(restored_samples)
    
    # Concatenate all rows
    full_grid = torch.cat(grid_images, dim=0)
    
    # Create grid layout: 3 rows x num_samples columns
    grid = vutils.make_grid(full_grid, nrow=num_samples, padding=2, normalize=False)
    
    # Convert to PIL image and save
    grid_np = grid.cpu().detach().numpy().transpose(1, 2, 0)
    grid_np = np.clip(grid_np * 255, 0, 255).astype(np.uint8)
    grid_pil = Image.fromarray(grid_np)
    
    # Save only the stitched comparison image
    grid_pil.save(os.path.join(save_path, f'epoch_{epoch}_batch_{batch_idx}_comparison.png'))

if __name__ == "__main__":
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    start_epoch = 1

    parser = argparse.ArgumentParser(description='Image Deraining')
    parser.add_argument('--train_dir', default='/home3/shpb49/Data/UIR/Dataset/for_SFNet_train/train', type=str,
                        help='Directory of train images')
    parser.add_argument('--val_dir', default='/home3/shpb49/Data/UIR/Dataset/for_SFNet_train/valid', type=str,
                        help='Directory of validation images')
    parser.add_argument('--model_save_dir', default='./checkpoints/', type=str, help='Path to save weights')
    parser.add_argument('--pretrain_weights', default='./checkpoints/model_best.pth', type=str,
                        help='Path to pretrain-weights')
    parser.add_argument('--mode', default='UIESNN', type=str)
    parser.add_argument('--session', default='DID-Data_new', type=str, help='session')
    parser.add_argument('--patch_size_train', default=64, type=int, help='training patch size')
    parser.add_argument('--patch_size_test', default=64, type=int, help='val patch size')
    parser.add_argument('--num_epochs', default=2000, type=int, help='num_epochs')
    parser.add_argument('--batch_size', default=12, type=int, help='batch_size')
    parser.add_argument('--val_epochs', default=10, type=int, help='val_epochs')
    parser.add_argument('--lr', default=1e-3, type=int, help='LearningRate')
    parser.add_argument('--min_lr', default=1e-7, type=int, help='min_LearningRate')
    parser.add_argument('--warmup_epochs', default=3, type=int, help='warmup_epochs')
    parser.add_argument('--clip_grad', default=1.0, type=float, help='clip_grad')
    parser.add_argument('--use_amp', default=False, type=bool, help='use_amp')
    parser.add_argument('--num_workers', default=2, type=int, help='num_workers')
    args = parser.parse_args()

    start_lr = args.lr
    end_lr = args.min_lr
    clip_grad = args.clip_grad
    use_amp = args.use_amp
    mode = args.mode
    session = args.session
    patch_size_train = args.patch_size_train
    patch_size_test = args.patch_size_test
    model_dir = os.path.join(args.model_save_dir, mode, 'models', session)
    utils.mkdir(model_dir)
    
    # Create directories for saving sample images
    sample_dir = os.path.join(args.model_save_dir, mode, 'samples', session)
    utils.mkdir(sample_dir)
    train_sample_dir = os.path.join(sample_dir, 'train_samples')
    val_sample_dir = os.path.join(sample_dir, 'val_samples')
    utils.mkdir(train_sample_dir)
    utils.mkdir(val_sample_dir)
    train_dir = args.train_dir
    val_dir = args.val_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    val_epochs = args.val_epochs
    num_workers = args.num_workers

    ######### Model ###########
    model_restoration = model
    model_restoration.cuda()

    functional.set_step_mode(model_restoration, step_mode='m')
    functional.set_backend(model_restoration, backend='cupy')

    # print number of model
    # get_parameter_number(model_restoration)
    # device_ids = 0
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    optimizer = optim.AdamW(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = args.warmup_epochs

    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)

    # scheduler_cosine = optim.lr_scheduler.StepLR(step_size=50, gamma=0.8,
    #                                       optimizer=optimizer)  ####step_size epoch, best_epoch 445

    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)

    # scheduler.step()
    RESUME = False
    Pretrain = False
    model_pre_dir = 'checkpoints/UIESNN/models/DID-Data'
    ######### Pretrain ###########
    if Pretrain:
        utils.load_checkpoint(model_restoration, model_pre_dir)

        print('------------------------------------------------------------------------------')
        print("==> Retrain Training with: " + model_pre_dir)
        print('------------------------------------------------------------------------------')

    ######### Resume ###########
    if RESUME:
        path_chk_rest = utils.get_last_path(model_pre_dir, '_last.pth')
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)
        # model_restoration.load_state_dict(torch.load(model_pre_dir))
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    ######### Loss ###########
    criterion_l1 = nn.L1Loss().cuda()
    criterion_ssim = utils.SSIM().cuda()
    criterion_psnr = PSNRLoss().cuda()
    ######### DataLoaders ###########

    dataset_train = Dataload(data_dir=train_dir, patch_size=patch_size_train)
    train_loader = DataLoader(dataset=dataset_train, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False,
                              pin_memory=True)

    dataset_val = Dataload(data_dir=val_dir, patch_size=patch_size_test)
    val_loader = DataLoader(dataset=dataset_val, num_workers=1, batch_size=batch_size, shuffle=False, drop_last=False,
                            pin_memory=True)
    # train_dataset = get_training_data(train_dir, {'patch_size': patch_size_train})
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                           drop_last=False,
    #                           pin_memory=True)
    #
    # val_dataset = get_validation_data(val_dir, {'patch_size': patch_size_test})
    # val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False,
    #                         pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
    print('===> Loading datasets')
    
    # Calculate model parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model_restoration)
    
    # Print hyperparameters
    print('=' * 80)
    print('HYPERPARAMETERS:')
    print('=' * 80)
    print(f'Mode: {mode}')
    print(f'Session: {session}')
    print(f'Training Directory: {train_dir}')
    print(f'Validation Directory: {val_dir}')
    print(f'Model Save Directory: {model_dir}')
    print(f'Sample Save Directory: {sample_dir}')
    print(f'Number of Epochs: {num_epochs}')
    print(f'Batch Size: {batch_size}')
    print(f'Validation Frequency: Every {val_epochs} epochs')
    print(f'Patch Size (Train): {patch_size_train}')
    print(f'Patch Size (Test): {patch_size_test}')
    print(f'Learning Rate: {start_lr}')
    print(f'Minimum Learning Rate: {end_lr}')
    print(f'Warmup Epochs: {warmup_epochs}')
    print(f'Gradient Clipping: {clip_grad}')
    print(f'Mixed Precision (AMP): {use_amp}')
    print(f'Number of Workers: {num_workers}')
    print(f'Number of GPUs: {len(device_ids)}')
    print(f'Device IDs: {device_ids}')
    print(f'Model Parameters: {total_params:,} ({total_params/1e6:.2f}M)')
    print('=' * 80)

    best_psnr = 0
    best_epoch = 0
    writer = SummaryWriter(model_dir)
    iter = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1
        train_psnr_val_rgb = []
        scaled_loss = 0
        model_restoration.train()
        # scheduler.step()
        for i, data in enumerate(tqdm(train_loader, unit='img'), 0):
            for param in model_restoration.parameters():
                param.grad = None
            target_ = data[1].cuda()
            input_ = data[0].cuda()
            
            # Model now returns three outputs
            output_level3, output_level2, restored = model_restoration(input_)

            # Build multi-scale targets/preds to match example.py:
            # L1 at 1/4, 1/2, 1x plus FFT loss at the same scales.
            target_l2 = torch.nn.functional.interpolate(target_, scale_factor=0.5, mode='bilinear', align_corners=False)
            target_l4 = torch.nn.functional.interpolate(target_, scale_factor=0.25, mode='bilinear', align_corners=False)
            pred_l2 = torch.nn.functional.interpolate(output_level2, scale_factor=0.5, mode='bilinear', align_corners=False)
            pred_l4 = torch.nn.functional.interpolate(output_level3, scale_factor=0.25, mode='bilinear', align_corners=False)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    # Content (L1) loss
                    l1_level3 = criterion_l1(pred_l4, target_l4)
                    l1_level2 = criterion_l1(pred_l2, target_l2)
                    l1_final = criterion_l1(restored, target_)
                    loss_content = l1_level3 + l1_level2 + l1_final

                    # SSIM loss (match the same multi-scale pairs)
                    ssim_level3 = criterion_ssim(pred_l4, target_l4)
                    ssim_level2 = criterion_ssim(pred_l2, target_l2)
                    ssim_final = criterion_ssim(restored, target_)
                    loss_ssim = (1 - ssim_level3) + (1 - ssim_level2) + (1 - ssim_final)

                    # FFT loss (L1 on stacked real/imag)
                    target_fft_l4 = torch.fft.fft2(target_l4, dim=(-2, -1))
                    target_fft_l4 = torch.stack((target_fft_l4.real, target_fft_l4.imag), -1)
                    pred_fft_l4 = torch.fft.fft2(pred_l4, dim=(-2, -1))
                    pred_fft_l4 = torch.stack((pred_fft_l4.real, pred_fft_l4.imag), -1)

                    target_fft_l2 = torch.fft.fft2(target_l2, dim=(-2, -1))
                    target_fft_l2 = torch.stack((target_fft_l2.real, target_fft_l2.imag), -1)
                    pred_fft_l2 = torch.fft.fft2(pred_l2, dim=(-2, -1))
                    pred_fft_l2 = torch.stack((pred_fft_l2.real, pred_fft_l2.imag), -1)

                    target_fft = torch.fft.fft2(target_, dim=(-2, -1))
                    target_fft = torch.stack((target_fft.real, target_fft.imag), -1)
                    pred_fft = torch.fft.fft2(restored, dim=(-2, -1))
                    pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)

                    fft_level3 = criterion_l1(pred_fft_l4, target_fft_l4)
                    fft_level2 = criterion_l1(pred_fft_l2, target_fft_l2)
                    fft_final = criterion_l1(pred_fft, target_fft)
                    loss_fft = fft_level3 + fft_level2 + fft_final

                    loss = loss_content + 0.1 * loss_fft + loss_ssim
                scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                functional.reset_net(model_restoration)
            else:
                # Content (L1) loss
                l1_level3 = criterion_l1(pred_l4, target_l4)
                l1_level2 = criterion_l1(pred_l2, target_l2)
                l1_final = criterion_l1(restored, target_)
                loss_content = l1_level3 + l1_level2 + l1_final

                # SSIM loss (match the same multi-scale pairs)
                ssim_level3 = criterion_ssim(pred_l4, target_l4)
                ssim_level2 = criterion_ssim(pred_l2, target_l2)
                ssim_final = criterion_ssim(restored, target_)
                loss_ssim = (1 - ssim_level3) + (1 - ssim_level2) + (1 - ssim_final)

                # FFT loss (L1 on stacked real/imag)
                target_fft_l4 = torch.fft.fft2(target_l4, dim=(-2, -1))
                target_fft_l4 = torch.stack((target_fft_l4.real, target_fft_l4.imag), -1)
                pred_fft_l4 = torch.fft.fft2(pred_l4, dim=(-2, -1))
                pred_fft_l4 = torch.stack((pred_fft_l4.real, pred_fft_l4.imag), -1)

                target_fft_l2 = torch.fft.fft2(target_l2, dim=(-2, -1))
                target_fft_l2 = torch.stack((target_fft_l2.real, target_fft_l2.imag), -1)
                pred_fft_l2 = torch.fft.fft2(pred_l2, dim=(-2, -1))
                pred_fft_l2 = torch.stack((pred_fft_l2.real, pred_fft_l2.imag), -1)

                target_fft = torch.fft.fft2(target_, dim=(-2, -1))
                target_fft = torch.stack((target_fft.real, target_fft.imag), -1)
                pred_fft = torch.fft.fft2(restored, dim=(-2, -1))
                pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)

                fft_level3 = criterion_l1(pred_fft_l4, target_fft_l4)
                fft_level2 = criterion_l1(pred_fft_l2, target_fft_l2)
                fft_final = criterion_l1(pred_fft, target_fft)
                loss_fft = fft_level3 + fft_level2 + fft_final

                loss = 0.5 * loss_content + 0.1 * loss_fft + loss_ssim
                
                loss.backward()
                scaled_loss += loss.item()
                # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                optimizer.step()
                functional.reset_net(model_restoration)
            torch.cuda.synchronize()
            epoch_loss += loss.item()
            iter += 1
            for res, tar in zip(restored, target_):
                train_psnr_val_rgb.append(utils.torchPSNR(res, tar))
            psnr_train = torch.stack(train_psnr_val_rgb).mean().item()

            writer.add_scalar('loss/iter_loss', loss.item(), iter)
            writer.add_scalar('loss/iter_loss_content', loss_content.item(), iter)
            writer.add_scalar('loss/iter_loss_fft', loss_fft.item(), iter)
            writer.add_scalar('loss/iter_loss_ssim', loss_ssim.item(), iter)
            writer.add_scalar('metrics/iter_ssim_final', ssim_final.item(), iter)
            writer.add_scalar('loss/iter_l1_level3_l4', l1_level3.item(), iter)
            writer.add_scalar('loss/iter_l1_level2_l2', l1_level2.item(), iter)
            writer.add_scalar('loss/iter_l1_final', l1_final.item(), iter)
            writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)
            writer.add_scalar('lr/epoch_loss', scheduler.get_lr()[0], epoch)
            
            # Save sample training images every 10 epochs
            if epoch % 10 == 0 and i == 0:  # Save only first batch of every 10th epoch
                save_sample_images(input_, target_, restored, train_sample_dir, epoch, i)
        #### Evaluation ####
        if epoch % val_epochs == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            psnr_val_level3 = []
            psnr_val_level2 = []
            val_sample_saved = False  # Flag to save only one sample per validation
            for ii, data_val in enumerate(tqdm(val_loader, unit='img'), 0):
                target = data_val[1].cuda()
                input_ = data_val[0].cuda()

                with torch.no_grad():
                    # Model now returns three outputs
                    output_level3, output_level2, restored = model_restoration(input_)
                functional.reset_net(model_restoration)

                # Calculate PSNR for all three outputs
                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))
                for res, tar in zip(output_level3, target):
                    psnr_val_level3.append(utils.torchPSNR(res, tar))
                for res, tar in zip(output_level2, target):
                    psnr_val_level2.append(utils.torchPSNR(res, tar))
                
                # Save sample validation images (only first batch)
                if not val_sample_saved:
                    save_sample_images(input_, target, restored, val_sample_dir, epoch, ii)
                    val_sample_saved = True

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            psnr_val_level3_mean = torch.stack(psnr_val_level3).mean().item()
            psnr_val_level2_mean = torch.stack(psnr_val_level2).mean().item()
            
            writer.add_scalar('val/psnr_final', psnr_val_rgb, epoch)
            writer.add_scalar('val/psnr_level3', psnr_val_level3_mean, epoch)
            writer.add_scalar('val/psnr_level2', psnr_val_level2_mean, epoch)
            
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(model_restoration.state_dict(), os.path.join(model_dir, "model_best.pth"))

            print("[epoch %d Training PSNR: %.4f --- best_epoch %d Test_PSNR %.4f (L3: %.4f, L2: %.4f)]" % 
                  (epoch, psnr_train, best_epoch, best_psnr, psnr_val_level3_mean, psnr_val_level2_mean))
        if epoch % 50 == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_last.pth"))
        scheduler.step()
        print("-" * 150)
        print(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tTrain_PSNR: {:.4f}\tSSIM: {:.4f}\tLearningRate {:.8f}\tTest_PSNR: {:.4f}".format(
                epoch, time.time() - epoch_start_time, loss.item(), psnr_train, ssim_final.item(), scheduler.get_lr()[0],
                best_psnr, ))
        print("-" * 150)
    writer.close()
