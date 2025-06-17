import torch
from train_emotion_model import EmotionCNN

# Load trained model
model = EmotionCNN()
model.load_state_dict(torch.load(r"models\emotion_cnn.pth", map_location="cpu"))
model.eval()

# Dummy input (batch of 1 grayscale image, 1x48x48)
dummy_input = torch.randn(1, 1, 48, 48)

# Export to ONNX
torch.onnx.export(
    model, dummy_input,
    r"models\emotion_model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("✅ Exported to ONNX format!")
