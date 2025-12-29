"""
人脸检测、裁剪与标准化工具 (Face Detector, Cropper, and Standardization Utility).

功能:
1. 使用 InsightFace 检测人脸。
2. 将检测到的人脸区域裁剪并标准化为一个固定尺寸的正方形图像。
3. 如果裁剪区域因边界限制不是完美正方形，使用镜像填充 (reflect padding) 进行补偿，
   避免黑边对 ResNet 等模型识别效果的影响。
4. 支持处理单个图片文件或整个文件夹内的所有图片。
5. 结果保存到指定输出文件夹，文件名与原图一致。

使用说明:
# 1. 处理单个文件，输出到默认文件夹 cropped_faces，尺寸 112x112
#    python face_processor.py my_image.jpg

# 2. 批量处理整个文件夹，输出到 output_data，尺寸 224x224
#    python face_processor.py ./input_data/ -o ./output_data/ -s 224
"""
import argparse
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


class FaceProcessor:
    """人脸处理核心类，包含检测、裁剪和标准化功能。"""

    def __init__(self, model_name: str = 'buffalo_s', providers: Optional[list] = None, det_size: tuple = (160, 160)):
        """初始化人脸检测器。

        Args:
            model_name: InsightFace 模型名称 (默认: buffalo_s)
            providers: ONNX runtime providers (默认: 自动检测 CUDA)
            det_size: 检测尺寸 (较小尺寸如 160x160 可兼顾速度和精度)
        """
        self.model_lock = threading.Lock()

        # 自动检测 CUDA 支持
        if providers is None:
            try:
                # 导入 torch 仅用于检查 CUDA 可用性
                import torch
                if torch.cuda.is_available():
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    print("🚀 FaceProcessor: 发现 CUDA，将使用 GPU 进行人脸检测")
                else:
                    providers = ['CPUExecutionProvider']
                    print("⚠️  FaceProcessor: CUDA 不可用，使用 CPU")
            except ImportError:
                providers = ['CPUExecutionProvider']
                print("⚠️  FaceProcessor: PyTorch 或 CUDA 库未找到，使用 CPU")

        self.model = FaceAnalysis(name=model_name, providers=providers)
        # 尝试使用 GPU (ctx_id=0) 或回退到 CPU (ctx_id=-1)
        try:
            self.model.prepare(ctx_id=0, det_size=det_size)
            print(f"✓ FaceProcessor 初始化成功，检测尺寸={det_size}, ctx_id=0 (GPU)")
        except Exception as e:
            print(f"⚠️  GPU 初始化失败 ({e})，回退到 CPU")
            self.model.prepare(ctx_id=-1, det_size=det_size)
            print(f"✓ FaceProcessor 初始化成功，检测尺寸={det_size}, ctx_id=-1 (CPU)")

    def detect(self, image: Image.Image, max_num: int = 1):
        """检测图像中的人脸。"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)
        # InsightFace 期望 BGR 格式 (OpenCV 约定), RGB -> BGR
        img_array = img_array[:, :, ::-1].copy()

        with self.model_lock:
            faces = self.model.get(img_array, max_num=max_num)

        del img_array
        return faces

    @staticmethod
    def crop_and_pad_face(image: Image.Image, bbox: np.ndarray, target_size: Tuple[int, int],
                          scale: float = 1.3) -> Image.Image:
        """
        裁剪人脸区域，确保为正方形，使用镜像填充避免黑边影响模型效果。

        Args:
            image: PIL Image
            bbox: 人脸边界框 [x1, y1, x2, y2]
            target_size: 最终输出的标准化尺寸 (例如 (112, 112))
            scale: 裁剪区域的缩放因子 (默认: 1.3)

        Returns:
            标准化后的 PIL Image
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        w, h = image.size

        # 1. 计算中心点和目标边长
        box_w = x2 - x1
        box_h = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 使用较长边作为裁剪正方形的边长，并应用缩放因子
        side_len = int(max(box_w, box_h) * scale)

        # 2. 计算理想正方形裁剪区域在原图上的坐标
        target_x1 = center_x - side_len // 2
        target_y1 = center_y - side_len // 2
        target_x2 = target_x1 + side_len
        target_y2 = target_y1 + side_len

        # 3. 计算需要填充的边距
        pad_left = max(0, -target_x1)
        pad_top = max(0, -target_y1)
        pad_right = max(0, target_x2 - w)
        pad_bottom = max(0, target_y2 - h)

        # 4. 如果需要填充，先对原图进行镜像扩展
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            # 使用 numpy 进行镜像填充 (reflect mode)
            img_array = np.array(image)
            
            # np.pad 的 reflect 模式会镜像复制边缘像素
            padded_array = np.pad(
                img_array,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='reflect'
            )
            padded_image = Image.fromarray(padded_array)
            
            # 调整裁剪坐标（因为图像已扩展）
            crop_x1 = target_x1 + pad_left
            crop_y1 = target_y1 + pad_top
            crop_x2 = crop_x1 + side_len
            crop_y2 = crop_y1 + side_len
            
            cropped = padded_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        else:
            # 无需填充，直接裁剪
            cropped = image.crop((target_x1, target_y1, target_x2, target_y2))

        # 5. 缩放到目标尺寸
        return cropped.resize(target_size, Image.Resampling.BILINEAR)

    def process_image(self, image_path: Path, output_dir: Path, target_size: Tuple[int, int], scale: float):
        """处理单张图片：检测、裁剪、标准化并保存。"""
        try:
            image = Image.open(image_path)

            # 1. 检测人脸
            # 仅处理第一张检测到的人脸
            faces = self.detect(image, max_num=1)

            if not faces:
                print(f"   [!] {image_path.name}: 未检测到人脸，跳过。")
                return

            # 2. 裁剪、补偿并标准化
            face = faces[0]
            # 这里使用了我们优化后的方法
            cropped_face = self.crop_and_pad_face(image, face.bbox, target_size, scale)

            # 3. 保存到目标文件夹
            output_path = output_dir / image_path.name

            # 确保目标尺寸和 RGB 模式
            if cropped_face.mode != 'RGB':
                cropped_face = cropped_face.convert('RGB')

            cropped_face.save(output_path)
            print(f"   [✓] {image_path.name}: 裁剪、标准化 ({target_size[0]}x{target_size[1]}) 并保存到 {output_path}")

        except Exception as e:
            # 打印详细错误信息
            print(f"   [✗] 处理 {image_path.name} 时发生错误: {e}")


def main():
    """主函数：处理命令行参数并执行人脸处理任务。"""
    parser = argparse.ArgumentParser(description="人脸裁剪、黑色补偿与尺寸标准化工具 (基于 ResNet 训练优化).",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "input_path",
        type=str,
        default="test.jpg",
        help="输入文件路径 (单张图片) 或文件夹路径 (多张图片)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="cropped_faces_output",
        help="输出文件夹路径，裁剪后的图片将保存到此目录。默认: cropped_faces_output"
    )
    parser.add_argument(
        "-s", "--size",
        type=int,
        default=112,
        help="最终标准化输出图片的正方形边长 (例如 112 表示 112x112)。默认: 112"
    )
    parser.add_argument(
        "-sc", "--scale",
        type=float,
        default=1.3,
        help="人脸裁剪区域的缩放因子 (越大包含越多背景，默认: 1.3，推荐值 1.2-1.5)。"
    )

    args = parser.parse_args()

    # 检查输入路径
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {args.input_path}")
        return

    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("人脸裁剪与标准化工具启动")
    print("=" * 60)
    print(f"目标输出文件夹: {output_dir.resolve()}")

    # 初始化处理器
    try:
        # 使用 det_size=(160, 160) 作为检测时的优化尺寸
        processor = FaceProcessor(det_size=(160, 160))
    except Exception as e:
        print(f"\n严重错误: 处理器初始化失败，请检查 InsightFace 依赖。错误: {e}")
        return

    target_size = (args.size, args.size)
    print(f"标准化输出尺寸: {target_size[0]}x{target_size[1]} 像素")
    print(f"裁剪缩放因子: {args.scale}\n")

    files_to_process = []

    if input_path.is_file():
        # 处理单个文件
        files_to_process.append(input_path)
    elif input_path.is_dir():
        # 处理文件夹
        print(f"开始批量处理文件夹: {input_path.resolve()}")
        # 查找常见图片格式
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            files_to_process.extend(input_path.glob(ext))

    if not files_to_process:
        print(f"在 {input_path} 中未找到任何图片文件 (jpg/jpeg/png/webp)。")
        return

    start_time = time.time()
    total_processed = 0

    # 循环处理文件
    for file_path in files_to_process:
        print(f"-> 正在处理: {file_path.name}")
        processor.process_image(file_path, output_dir, target_size, args.scale)
        total_processed += 1

    end_time = time.time()

    print("\n" + "=" * 60)
    print("✅ 处理完成总结")
    print("-" * 60)
    print(f"总处理文件数: {total_processed}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"结果已保存到目录: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == '__main__':
    # 注意：在本地运行时，直接执行即可。在某些在线环境中，您可能需要手动调用 main()
    main()