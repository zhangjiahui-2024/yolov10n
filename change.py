import os
import subprocess

# 模型文件路径
onnx_model_path = r"E:\大三\深度学习\新建文件夹\yolov10-main\runs\detect\train10\weights\best.onnx"
# 输出目录
output_dir = r"E:\大三\深度学习\新建文件夹\yolov10-main\runs\detect\train10\weights\openvino_model"
# 数据集 YAML 文件路径
data_yaml_path = r"E:\大三\深度学习\新建文件夹\yolov10-main\split_data\data.yaml"  # 你的 YAML 文件路径


# 1. 使用 OpenVINO 的 Model Optimizer 转换 ONNX 模型并进行 INT8 量化
# 调用命令行工具 mo 来转换模型，并进行 INT8 量化
def run_model_optimizer():
    command = [
        "mo",
        "--input_model", onnx_model_path,  # 输入的 ONNX 模型路径
        "--framework", "onnx",  # 输入框架：ONNX
        "--output_dir", output_dir,  # 输出目录
        "--quantize",  # 启用量化
        "--data_type", "int8",  # 设置量化为 INT8
        "--input", "images",  # 输入层名称（可根据模型调整）
        "--output", "output",  # 输出层名称（可根据模型调整）
        "--batch_size", "1",  # 批量大小
        "--weights", "weights",  # 权重层（可根据模型调整）
        "--data", data_yaml_path  # 指定数据集的 YAML 文件路径
    ]
    # 运行命令并等待结束
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("Model optimization and quantization completed successfully.")
        print(f"Quantized model saved to: {output_dir}")
        print(f"Generated files: {os.path.join(output_dir, 'best.xml')} and {os.path.join(output_dir, 'best.bin')}")
    else:
        print("Error in model optimization:", result.stderr)


# 2. 使用 OpenVINO 进行推理
from openvino.runtime import Core
import cv2
import numpy as np


def run_inference():
    # 加载量化后的 OpenVINO 模型
    model_path_xml = os.path.join(output_dir, "best.xml")  # 量化后的 XML 模型文件
    model_path_bin = os.path.join(output_dir, "best.bin")  # 量化后的 BIN 权重文件
    ie = Core()
    model = ie.read_model(model=model_path_xml, weights=model_path_bin)
    compiled_model = ie.compile_model(model=model, device_name="CPU")  # 使用 CPU 进行推理

    # 输入输出层
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # 加载并预处理图片
    image_path = r"E:\大三\深度学习\新建文件夹\yolov10-main\split_data\valid\images\2022329600041-1.jpg"  # 替换为你的图片路径
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_layer.shape[3], input_layer.shape[2]))  # 调整图片大小
    image_input = np.transpose(image_resized, (2, 0, 1))  # 转换为 (C, H, W) 形状
    image_input = image_input[np.newaxis, ...]  # 增加批次维度

    # 执行推理
    result = compiled_model([image_input])[output_layer]

    # 处理并显示结果（假设模型输出为框和类别信息，需根据具体模型调整）
    for detection in result[0]:  # 根据输出格式调整
        xmin, ymin, xmax, ymax, confidence, class_id = detection[:6]  # 提取信息
        if confidence > 0.5:  # 设置阈值
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(image, f"Class {int(class_id)}: {confidence:.2f}", (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示推理结果
    cv2.imshow("Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 运行转换和量化
run_model_optimizer()

# 运行推理
run_inference()
