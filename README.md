# yolov10n运行步骤  

已上传app.py、exportmodel.py、change.py、11test.py文件  
结合上传的模型，best.pt可以通过app.py运行，最终结果会在网页中可视化显示
best.pt模型可通过exportmodel.py、change.py进行转换与优化，先通过exportmodel.py将best.pt转化为best.onnx，再运行change.py文件，将模型转换为openvino形式，并量化为int8
