import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm # 导入tqdm用于显示进度条

# 导入MONAI的核心组件
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

# --- 1. 参数设置 ---
clean_image_path = r"F:\MONAI\train_g_data\ll\ground_truth_train_000_slice_001.png"
model_path = r"F:\MONAI\ldm\output\final_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像和噪声参数
IMG_SIZE = 176
noise_step_t = 10  # 我们将从这个时间步开始去噪
total_train_timesteps = 1000

# --- 2. 加载您训练好的DDPM (U-Net) 模型 ---
model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 128,),
    attention_levels=(False, True, False),
    num_res_blocks=2,
    num_head_channels=32
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✅ 模型加载成功！")

# --- 3. 创建DDPM调度器 ---
scheduler = DDPMScheduler(
    num_train_timesteps=total_train_timesteps,
    beta_start=0.0015,
    beta_end=0.0205,
    schedule="scaled_linear_beta"
)
# vvvvvvvvvvvvvvvvvvvv   【新增】设置调度器的时间步长   vvvvvvvvvvvvvvvvvvvv
# 这会生成一个从999到0的序列，供我们后续迭代使用
scheduler.set_timesteps(num_inference_steps=total_train_timesteps)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
print("✅ 调度器创建成功！")

# --- 4. 加载并预处理原始图像 ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

try:
    original_image_pil = Image.open(clean_image_path)
    original_image_tensor = transform(original_image_pil).unsqueeze(0).to(device)
    print(f"✅ 原始图像加载成功: {clean_image_path}")
except FileNotFoundError:
    print(f"❌ 错误：找不到原始图像文件，请检查路径：{clean_image_path}")
    exit()

# --- 5. 手动添加噪声 (前向过程) ---
noise = torch.randn_like(original_image_tensor)
add_noise_timesteps = torch.tensor([noise_step_t], device=device).long()
noisy_image_tensor = scheduler.add_noise(
    original_samples=original_image_tensor, noise=noise, timesteps=add_noise_timesteps
)
print(f"✅ 已向图像添加 t={noise_step_t} 的噪声。")


# --- 6. 使用模型进行【多步】去噪 (逆向过程) ---
# vvvvvvvvvvvvvvvvvvvvvv   【已修改为循环去噪】   vvvvvvvvvvvvvvvvvvvvvv
# 从我们加噪后的图片开始，作为去噪的起点
denoising_image_tensor = noisy_image_tensor

# 筛选出从 t=500 开始到 t=0 的所有时间步
timesteps_to_iterate = scheduler.timesteps[total_train_timesteps - noise_step_t:]

print(f"🚀 开始从 t={noise_step_t} 进行 {len(timesteps_to_iterate)} 步去噪...")

with torch.no_grad():
    # 使用tqdm创建进度条
    for t in tqdm(timesteps_to_iterate):
        # 将时间步t转换为张量
        timestep_tensor = torch.tensor([t], device=device).long()
        
        # 1. 预测噪声
        predicted_noise = model(x=denoising_image_tensor, timesteps=timestep_tensor)
        
        # 2. 计算上一步的图像 (去噪)
        denoising_output = scheduler.step(
            model_output=predicted_noise,
            timestep=t,
            sample=denoising_image_tensor
        )
        
        # 3. 更新图像，用于下一次迭代
        denoising_image_tensor = denoising_output[0]

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
print("✅ 已完成多步去噪。")


# --- 7. 显示结果 ---
def tensor_to_pil(tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = tensor.detach().cpu().squeeze()
    img_np = tensor.numpy()
    img = Image.fromarray((img_np * 255).astype(np.uint8))
    return img

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(tensor_to_pil(original_image_tensor), cmap='gray')
axes[0].set_title('Original Clean Image', fontsize=16)
axes[0].axis('off')

axes[1].imshow(tensor_to_pil(noisy_image_tensor), cmap='gray')
axes[1].set_title(f'Noisy Image (Start at t={noise_step_t})', fontsize=16)
axes[1].axis('off')

axes[2].imshow(tensor_to_pil(denoising_image_tensor), cmap='gray')
axes[2].set_title(f'Denoised Image (After {len(timesteps_to_iterate)} Steps)', fontsize=16)
axes[2].axis('off')

plt.tight_layout()
plt.show()
