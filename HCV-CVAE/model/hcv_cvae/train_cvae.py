import os 
import random
import shutil
import yaml
import math
import gc
from attrdict import AttrMap
import time

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
import torch.cuda
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from dataload.init import getdata
from .network import VAE
import utils.utils as utils
from utils.utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport
from .edg_loss import edge_loss
from .kl_annealing import get_kl_annealing
from .safe_kl_annealing import get_safe_kl_annealing, get_predefined_safe_annealing

# 激进的内存优化设置
num_samples = 1  # 若要多样本评估，设置为 >1

# -----------------------------
# Helper utilities (PyTorch1.7 safe)
# -----------------------------
def safe_nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9):
    """
    Replace NaN with `nan`, +inf with posinf, -inf with neginf.
    Implemented with basic ops to be compatible with PyTorch 1.7.
    """
    if not torch.is_tensor(x):
        return x
    # Replace NaNs
    x = torch.where(torch.isnan(x), torch.full_like(x, nan), x)
    # Replace +inf / -inf
    posinf_mask = torch.isinf(x) & (x > 0)
    neginf_mask = torch.isinf(x) & (x < 0)
    x = torch.where(posinf_mask, torch.full_like(x, posinf), x)
    x = torch.where(neginf_mask, torch.full_like(x, neginf), x)
    return x

def clamp_tensor(x, min_val=-1e4, max_val=1e4):
    """Clamp tensor to a reasonable range after NaN/Inf sanitization."""
    return torch.clamp(x, min=min_val, max=max_val)

def save_bad_batch(config, real_a, real_b, M, epoch, batch_idx, reason="nan"):
    """Save problematic batch tensors and a small preview for inspection."""
    bad_dir = os.path.join(config.out_dir, "bad_batches")
    os.makedirs(bad_dir, exist_ok=True)
    name = f"epoch{epoch:03d}_batch{batch_idx:04d}_{reason}"
    pt_path = os.path.join(bad_dir, name + ".pt")
    try:
        torch.save({'real_a': real_a.cpu(), 'real_b': real_b.cpu(), 'M': M.cpu()}, pt_path)
    except Exception as e:
        print(f"⚠️ Failed saving bad batch tensors: {e}")

    # try saving preview images for the first sample in the batch
    try:
        preview_dir = os.path.join(bad_dir, "preview")
        os.makedirs(preview_dir, exist_ok=True)
        # save_image expects tensor in expected format; if it fails, ignore
        save_image(real_a[0], os.path.join(preview_dir, name + "_input.png"))
        save_image(real_b[0], os.path.join(preview_dir, name + "_target.png"))
    except Exception as e:
        # preview optional
        pass

# -----------------------------
# Main train function (enhanced)
# -----------------------------
def train(config):
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')

    train_dataset, validation_dataset = getdata(config)
    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads,
                                      batch_size=config.batchsize, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads,
                                        batch_size=config.validation_batchsize, shuffle=False)
    
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = VAE(in_channels=config.in_ch)
    print('Total params: %.2fM' % (sum(p.numel() for p in gen.parameters())/1000000.0))

    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    gen = gen.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    criterionSoftplus = criterionSoftplus.cuda()
    L1_Loss = nn.SmoothL1Loss(beta=0.5).cuda()

    # Initialize perceptual loss functions
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex', version='0.1', spatial=True).cuda()
        print('===> LPIPS perceptual loss initialized')
    except Exception as e:
        print('===> LPIPS not available, skipping perceptual loss:', e)
        lpips_fn = None
    
    # VGG perceptual loss (optional)
    try:
        from model.msda.perceptual import LossNetworkL1
        vgg_loss_fn = LossNetworkL1().cuda()
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        print('===> VGG perceptual loss initialized')
    except Exception as e:
        print('===> VGG perceptual loss not available:', e)
        vgg_loss_fn = None
        imagenet_mean = None
        imagenet_std = None

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    # Initialize KL annealing strategy
    kl_history = []  # Store KL weight and divergence history
    try:
        with open('model/cvae/kl_config.yaml', 'r') as f:
            kl_config = yaml.safe_load(f)
        
        strategy = kl_config.get('strategy', 'linear')
        strategy_params = kl_config.get(strategy, {})

        if strategy == 'safe_linear':
            kl_annealer = get_safe_kl_annealing('safe_linear', **strategy_params)
        elif strategy == 'conservative':
            kl_annealer = get_predefined_safe_annealing('conservative')
        elif strategy == 'moderate':
            kl_annealer = get_predefined_safe_annealing('moderate')
        elif strategy == 'adaptive':
            kl_annealer = get_predefined_safe_annealing('adaptive')
        else:
            kl_annealer = get_kl_annealing(strategy, **strategy_params)
        
        print(f'===> Using KL annealing strategy: {strategy} with params: {strategy_params}')
    except Exception as e:
        print(f'===> Error loading KL config: {e}')
        kl_annealer = get_predefined_safe_annealing('conservative')
        print(f'===> Using fallback conservative KL annealing strategy')

    print('===> begin')
    start_time=time.time()
    
    # 创建时间记录文件
    time_log_file = os.path.join(config.out_dir, 'training_time_log.txt')
    with open(time_log_file, 'w', encoding='utf-8') as f:
        f.write("CVAE Training Time Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {config.epoch}\n")
        f.write(f"Batch size: {config.batchsize}\n")
        f.write(f"Learning rate: {config.lr}\n")
        f.write("=" * 50 + "\n\n")
    
    # 启用梯度检查点以节省内存（如果模型支持）
    if hasattr(gen, 'gradient_checkpointing_enable'):
        try:
            gen.gradient_checkpointing_enable()
        except Exception:
            pass
    
    # 设置更激进的内存管理
    torch.cuda.empty_cache()
    gc.collect()
    
    # 记录无效KL散度的计数器
    invalid_kl_count = 0
    max_invalid_kl = 100  # 最大允许的无效KL散度数量
    
    # main
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        gen.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            # 在每个批次开始时清理内存
            if iteration % 5 == 1:
                torch.cuda.empty_cache()
                gc.collect()
            
            # load batch and move to GPU (ensure float)
            try:
                real_a_cpu, real_b_cpu, M_cpu = batch[0], batch[1], batch[2]
            except Exception as e:
                print(f"⚠️  Malformed batch at epoch {epoch}, iter {iteration}: {e}")
                continue

            # convert to float and to cuda; keep copies for saving if needed
            real_a = Variable(real_a_cpu.float()).cuda()
            real_b = Variable(real_b_cpu.float()).cuda()
            M = Variable(M_cpu.float()).cuda()

            # quick input sanitation
            if torch.isnan(real_a).any() or torch.isinf(real_a).any() or torch.isnan(real_b).any() or torch.isinf(real_b).any():
                print(f"⚠️ Invalid input detected at Epoch {epoch}, Batch {iteration} - saving and skipping")
                save_bad_batch(config, real_a, real_b, M, epoch, iteration, reason="input_nan")
                opt_gen.zero_grad()
                continue

            opt_gen.zero_grad()

            # --- Forward pass ---
            try:
                (mean, log_var), cloud_removal = gen.forward(real_a)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"🚨 CUDA OOM at Epoch {epoch}, Batch {iteration}: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    opt_gen.zero_grad()
                    continue
                else:
                    raise e

            # sanitize outputs early
            mean = safe_nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
            log_var = safe_nan_to_num(log_var, nan=0.0, posinf=10.0, neginf=-10.0)
            cloud_removal = safe_nan_to_num(cloud_removal, nan=0.0, posinf=1e6, neginf=-1e6)

            # quick post-forward checks
            if torch.isnan(cloud_removal).any() or torch.isinf(cloud_removal).any():
                print(f"⚠️  Warning: Model output contains NaN/Inf at Epoch {epoch}, Batch {iteration}")
                save_bad_batch(config, real_a, real_b, M, epoch, iteration, reason="out_nan")
                opt_gen.zero_grad()
                continue
            
            if torch.isnan(mean).any() or torch.isinf(mean).any():
                print(f"⚠️  Warning: Mean contains NaN/Inf at Epoch {epoch}, Batch {iteration}")
                save_bad_batch(config, real_a, real_b, M, epoch, iteration, reason="mean_nan")
                opt_gen.zero_grad()
                continue
                    
            if torch.isnan(log_var).any() or torch.isinf(log_var).any():
                print(f"⚠️  Warning: Log variance contains NaN/Inf at Epoch {epoch}, Batch {iteration}")
                save_bad_batch(config, real_a, real_b, M, epoch, iteration, reason="logvar_nan")
                opt_gen.zero_grad()
                continue

            # Clamp predictions to expected range if known (commonly [-1,1] for normalized images)
            cloud_removal = clamp_tensor(cloud_removal, -10.0, 10.0)
            real_b = clamp_tensor(real_b, -10.0, 10.0)

            # --- Multiple samples with aggressive memory optimization ---
            if num_samples > 1:
                for i in range(num_samples-1):
                    try:
                        # explicit disable autocast to avoid fp16 instability on 1.7
                        with torch.cuda.amp.autocast(enabled=False):
                            (m, l), c = gen.forward(real_a)
                            m = safe_nan_to_num(m, nan=0.0, posinf=1e6, neginf=-1e6)
                            l = safe_nan_to_num(l, nan=0.0, posinf=10.0, neginf=-10.0)
                            c = safe_nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6)
                            
                            # 检查输出有效性
                            if torch.isnan(c).any() or torch.isinf(c).any():
                                print(f"⚠️  Warning: Sample {i+1} output contains NaN/Inf at Epoch {epoch}, Batch {iteration}")
                                break
                                
                            cloud_removal = cloud_removal + c
                            mean = mean + m
                            log_var = log_var + l
                        
                        # 释放中间变量
                        del m, l, c
                        
                        if i % 2 == 0:
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"🚨 CUDA OOM during sampling {i+1}: {e}")
                            break
                        else:
                            raise e
                
                # take expectation
                cloud_removal = cloud_removal / float(num_samples)
                mean = mean / float(num_samples)
                log_var = log_var / float(num_samples)
                gc.collect()
                torch.cuda.empty_cache()

            # 计算损失并检查数值稳定性
            try:
                # L1 loss
                l1_loss = L1_Loss(cloud_removal, real_b)
                if torch.isnan(l1_loss) or torch.isinf(l1_loss):
                    print(f"⚠️  Warning: L1 loss is {l1_loss} at Epoch {epoch}, Batch {iteration}")
                    save_bad_batch(config, real_a, real_b, M, epoch, iteration, reason="l1_nan")
                    opt_gen.zero_grad()
                    continue

                # KL: clamp log_var BEFORE exp to avoid overflow
                log_var_safe = torch.clamp(log_var, min=-10.0, max=10.0)
                # compute exp safely
                try:
                    exp_logvar = torch.exp(log_var_safe)
                except Exception:
                    exp_logvar = torch.clamp(torch.exp(torch.clamp(log_var_safe, max=10.0)), max=1e6)
                kl_div = -0.5 * torch.sum(1.0 + log_var_safe - mean.pow(2) - exp_logvar)
                # normalize kl by batch size to keep magnitude reasonable
                try:
                    bs = float(real_a.size(0))
                    kl_div = kl_div / (bs + 1e-8)
                except Exception:
                    pass

                # sanitize kl_div
                kl_div = safe_nan_to_num(kl_div, nan=0.0, posinf=1e9, neginf=0.0)
                if torch.isnan(kl_div) or torch.isinf(kl_div):
                    invalid_kl_count += 1
                    print(f"⚠️  Invalid KL divergence: {float(kl_div)} , count: {invalid_kl_count}")
                    save_bad_batch(config, real_a, real_b, M, epoch, iteration, reason="kl_nan")
                    # if too many invalid KLs, fallback to tiny weight, otherwise skip batch
                    if invalid_kl_count > max_invalid_kl:
                        print("🚨  Too many invalid KL divergences, using emergency fallback")
                        kl_weight = 0.0001
                        kl_div = torch.tensor(0.0, device=cloud_removal.device)
                    else:
                        opt_gen.zero_grad()
                        continue
                else:
                    invalid_kl_count = 0
                
                EDGE_loss = edge_loss(cloud_removal, real_b, 'cuda')
                if torch.isnan(EDGE_loss) or torch.isinf(EDGE_loss):
                    print(f"⚠️  Warning: Edge loss invalid at Epoch {epoch}, Batch {iteration}")
                    save_bad_batch(config, real_a, real_b, M, epoch, iteration, reason="edge_nan")
                    opt_gen.zero_grad()
                    continue
                
                # Use KL annealing strategy (external annealer returns float or tensor)
                try:
                    kl_weight = kl_annealer.get_weight(epoch, kl_div)
                except Exception:
                    kl_weight = 0.0001
                # sanitize weight
                try:
                    if isinstance(kl_weight, torch.Tensor):
                        kl_weight = float(kl_weight.item())
                    else:
                        kl_weight = float(kl_weight)
                except Exception:
                    kl_weight = 0.0001
                if math.isnan(kl_weight) or math.isinf(kl_weight):
                    print(f"⚠️  Warning: KL weight is {kl_weight} at Epoch {epoch}, Batch {iteration}")
                    kl_weight = 0.0001
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"🚨 CUDA OOM during loss computation: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    opt_gen.zero_grad()
                    continue
                else:
                    raise e
            
            # Initialize loss components with memory optimization
            lpips_loss = 0
            vgg_perc = 0
            fft_loss = 0
            
            # 只在每5个批次计算感知损失以节省内存
            if iteration % 5 == 0:
                try:
                    # LPIPS spatial perceptual loss (if available)
                    if lpips_fn is not None:
                        # ensure inputs in expected range and finite
                        cr = clamp_tensor(safe_nan_to_num(cloud_removal, nan=0.0, posinf=1e6, neginf=-1e6), -1.5, 1.5)
                        rb = clamp_tensor(safe_nan_to_num(real_b, nan=0.0, posinf=1e6, neginf=-1e6), -1.5, 1.5)
                        lpips_map = lpips_fn(cr, rb)
                        lpips_loss = lpips_map.mean()
                        if torch.isnan(lpips_loss) or torch.isinf(lpips_loss):
                            print(f"⚠️  Warning: LPIPS loss is {lpips_loss}")
                            lpips_loss = 0
                        del lpips_map  # 立即清理

                    # VGG perceptual loss (if available)
                    if vgg_loss_fn is not None and imagenet_mean is not None and imagenet_std is not None:
                        pred_rgb = cloud_removal[:, :3, :, :]
                        gt_rgb = real_b[:, :3, :, :]
                        # map from [-1,1] to [0,1] for VGG
                        pred_01 = (pred_rgb + 1.0) * 0.5
                        gt_01 = (gt_rgb + 1.0) * 0.5
                        # normalize to ImageNet stats
                        pred_vgg = (pred_01 - imagenet_mean) / imagenet_std
                        gt_vgg = (gt_01 - imagenet_mean) / imagenet_std
                        vgg_perc = vgg_loss_fn(pred_vgg, gt_vgg)
                        if torch.isnan(vgg_perc) or torch.isinf(vgg_perc):
                            print(f"⚠️  Warning: VGG loss is {vgg_perc}")
                            vgg_perc = 0
                        # 清理中间变量
                        del pred_rgb, gt_rgb, pred_01, gt_01, pred_vgg, gt_vgg

                    # FFT magnitude spectrum loss for high-frequency detail preservation
                    def fft_magnitude_loss(pred, target):
                        # Convert to grayscale for FFT computation
                        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
                        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
                        
                        # sanitize
                        pred_gray = safe_nan_to_num(pred_gray, nan=0.0, posinf=1e6, neginf=-1e6)
                        target_gray = safe_nan_to_num(target_gray, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        # Compute FFT (compatible with PyTorch 1.7)
                        pred_fft = torch.rfft(pred_gray, 2, onesided=False)
                        target_fft = torch.rfft(target_gray, 2, onesided=False)
                        
                        # Get magnitude spectrum (for rfft output: [B, C, H, W, 2] where last dim is [real, imag])
                        pred_mag = torch.sqrt(torch.clamp(pred_fft[..., 0]**2 + pred_fft[..., 1]**2, min=1e-8))
                        target_mag = torch.sqrt(torch.clamp(target_fft[..., 0]**2 + target_fft[..., 1]**2, min=1e-8))
                        
                        # L1 loss on magnitude spectrum
                        loss = F.l1_loss(pred_mag, target_mag)
                        
                        # 清理中间变量
                        del pred_gray, target_gray, pred_fft, target_fft, pred_mag, target_mag
                        return loss
                    
                    fft_loss = fft_magnitude_loss(cloud_removal, real_b)
                    
                    # 检查FFT损失是否有效
                    if torch.isnan(fft_loss) or torch.isinf(fft_loss):
                        print(f"⚠️  Warning: FFT loss is {fft_loss}")
                        fft_loss = 0
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"🚨 CUDA OOM during perceptual loss computation: {e}")
                        print("   Skipping perceptual losses for this batch")
                        lpips_loss = 0
                        vgg_perc = 0
                        fft_loss = 0
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        raise e

            # Combined loss with KL annealing
            try:
                Loss = l1_loss + kl_weight*kl_div + 0.18*EDGE_loss + 0.2*lpips_loss + 0.1*vgg_perc + 0.05*fft_loss
                
                # 检查总损失是否有效
                Loss = safe_nan_to_num(Loss, nan=0.0, posinf=1e9, neginf=0.0)
                if torch.isnan(Loss) or torch.isinf(Loss):
                    print(f"⚠️  Warning: Total loss is invalid at Epoch {epoch}, Batch {iteration}")
                    save_bad_batch(config, real_a, real_b, M, epoch, iteration, reason="loss_nan")
                    opt_gen.zero_grad()
                    continue
                
                # 反向传播
                Loss.backward()
                
                # 检查梯度是否含 NaN/Inf
                bad_grad = False
                for p in gen.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            bad_grad = True
                            break

                if bad_grad:
                    print(f"⚠️ NaN/Inf detected in gradients at Epoch {epoch}, Batch {iteration}. Saving batch and skipping optimizer step.")
                    save_bad_batch(config, real_a, real_b, M, epoch, iteration, reason="grad_nan")
                    opt_gen.zero_grad()
                    continue
                
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=0.5)  # 更严格的梯度裁剪
                
                # 优化器步骤
                opt_gen.step()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"🚨 CUDA OOM during backward pass: {e}")
                    print("   Clearing cache and skipping batch")
                    torch.cuda.empty_cache()
                    gc.collect()
                    opt_gen.zero_grad()
                    continue
                else:
                    # 保存触发异常的 batch 以便离线调试
                    save_bad_batch(config, real_a, real_b, M, epoch, iteration, reason="runtime_err")
                    raise e
            
            # 立即清理所有变量以释放内存
            try:
                del cloud_removal, mean, log_var, l1_loss, kl_div, EDGE_loss
                if 'lpips_loss' in locals():
                    del lpips_loss
                if 'vgg_perc' in locals():
                    del vgg_perc
                if 'fft_loss' in locals():
                    del fft_loss
                del Loss
            except:
                pass  # 忽略删除错误
            
            # 更频繁的内存清理
            if iteration % 3 == 0:  # 每3个批次清理一次
                gc.collect()
                torch.cuda.empty_cache()
            
            # 记录KL退火历史
            if iteration % 10 == 0:
                try:
                    kl_div_value = float(kl_div) if (not torch.isnan(kl_div)) else 0.0
                    kl_history.append({
                        'epoch': epoch,
                        'batch': iteration,
                        'kl_weight': kl_weight,
                        'kl_div': kl_div_value
                    })
                    print(f'Epoch {epoch}, Batch {iteration}: KL weight = {kl_weight:.6f}, KL div = {kl_div_value:.6f}')
                except Exception:
                    pass  # 忽略记录错误


        epoch_time = time.time() - epoch_start_time
        print('epoch', epoch, 'finished, use time', epoch_time)
        
        # 记录每个epoch的时间到文件
        with open(time_log_file, 'a', encoding='utf-8') as f:
            f.write(f"Epoch {epoch:3d}: {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)\n")
        
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        if epoch % config.snapshot_interval == 0 and epoch > 50:
            checkpoint(config, epoch, gen)

        logreport.save_lossgraph()
        validationreport.save_lossgraph()
        
        # Save KL annealing history
        if kl_history:
            try:
                import json
                kl_history_file = os.path.join(config.out_dir, 'kl_annealing_history.json')
                with open(kl_history_file, 'w') as f:
                    json.dump(kl_history, f, indent=2)
                print(f'KL annealing history saved to {kl_history_file}')
            except Exception as e:
                print(f"⚠️ Failed to save KL history: {e}")
    
    total_training_time = time.time() - start_time
    print('training time:', total_training_time)
    
    # 记录总训练时间到文件
    with open(time_log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)\n")
        f.write(f"Average time per epoch: {total_training_time/config.epoch:.2f} seconds\n")
        f.write("=" * 50 + "\n")
    
    print(f"Training time log saved to: {time_log_file}")
