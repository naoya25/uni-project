from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import cv2

# Stable Diffusion v2のモデル名
model_id = "stabilityai/stable-diffusion-2"

# ノイズスケジューラ
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# 重みのダウンロード & モデルのロード
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
# GPU使用。（CPUだと生成にかなり時間かかります。というかいつ終わるのか不明。）
pipe = pipe.to("cuda")

# 入力テキスト
# 訳：スタバで四苦八苦しながら論文を読んでいるロボット
prompt = "a robot struggling with reading through a scientific paper at Starbucks"
image = pipe(prompt).images[0]

cv2.imwrite('data/dst/lena_opencv_red.jpg', image)
