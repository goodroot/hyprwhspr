import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
print(model.recognize("2086-149220-0033.wav"))
