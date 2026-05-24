import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from visual_sentry_model import VisualSentry
import matplotlib.pyplot as plt
import os
import numpy as np


def process_image(image_path):
    # Load image natively in full RGB color
    img = Image.open(image_path).convert('RGB')

    # Convert to a NumPy array to perform vector mathematics
    img_array = np.array(img)  # Shape: (Height, Width, 3)

    # Senior Engineering Move: Take the minimum value along the channel axis (axis=2)
    # This automatically forces Red, Green, Blue, or Black ink to turn dark,
    # while leaving the clean white paper background completely bright white.
    universal_grayscale = np.min(img_array, axis=2).astype(np.uint8)

    # Convert back to PIL Image and apply our standard 64x64 rescaling
    processed_img = Image.fromarray(universal_grayscale).resize((64, 64))

    img_tensor = transforms.ToTensor()(processed_img)
    img_tensor = transforms.Normalize((0.5,), (0.5,))(img_tensor).unsqueeze(0)  # Shape: (1, 1, 64, 64)
    return img_tensor


def run_diagnostics(anchor_paths, suspect_path):
    model = VisualSentry(num_classes=47)
    if not os.path.exists("visualsentry_v1.pth"):
        print("Error: Run 'visual_sentry_train.py' first to initialize the network weights!")
        return

    model.load_state_dict(torch.load("visualsentry_v1.pth", map_location=torch.device('cpu')))
    model.eval()

    # Extract profiles from real white paper images
    anchor_vectors = []
    print("Enrolling Authentic Signature Profiles...")
    for path in anchor_paths:
        if os.path.exists(path):
            tensor = process_image(path)
            with torch.no_grad():
                vector = model(tensor, return_features=True)
                anchor_vectors.append(vector)
        else:
            print(f"Warning: Reference image {path} not found.")

    if not anchor_vectors:
        print("Error: Zero valid reference anchors found. Check file placement.")
        return

    mean_anchor_vector = torch.mean(torch.stack(anchor_vectors), dim=0)

    if not os.path.exists(suspect_path):
        print(f"Error: Suspect target file '{suspect_path}' is missing.")
        return

    suspect_tensor = process_image(suspect_path)

    with torch.no_grad():
        suspect_vector = model(suspect_tensor, return_features=True)
        logits = model(suspect_tensor)
        prob = F.softmax(logits, dim=1)
        conf, pred = torch.max(prob, 1)

    similarity = F.cosine_similarity(mean_anchor_vector, suspect_vector)

    print(f"\n" + "=" * 50)
    print(f"      VISUAL-SENTRY CONVOLUTIONAL DIAGNOSTIC REPORT")
    print(f"=" * 50)
    print(f"ENROLLED ANCHORS : {len(anchor_vectors)}")
    print(f"STYLOMETRIC DNA  : {similarity.item() * 100:.2f}%")
    print(f"-" * 50)

    # 92% Strict Threshold Check
    if similarity.item() > 0.92:
        print("FINAL VERDICT    : ✅ VALID BIOMETRIC MATCH")
    else:
        print("FINAL VERDICT    : ❌ FRAUD DETECTED - STYLOMETRIC VARIANCE")
    print(f"=" * 50)

    print("\nGenerating Convolutional Feature Map Visualization...")
    with torch.no_grad():
        # Capture spatial patterns at our new 64x64 scale resolution
        first_layer_features = model.pool1(model.relu1(model.conv1(suspect_tensor)))

    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    fig.suptitle("Visual-Sentry Spatial Activation Channels", fontsize=12, fontweight='bold')

    for i in range(16):
        ax = axes[i // 8, i % 8]
        f_map = first_layer_features[0, i].cpu().numpy()
        ax.imshow(f_map, cmap='magma')
        ax.axis('off')
        ax.set_title(f"Ch {i + 1}", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    my_anchors = ["real_1.png", "real_2.png", "real_3.png"]
    suspect = "test_image.png"
    run_diagnostics(my_anchors, suspect)