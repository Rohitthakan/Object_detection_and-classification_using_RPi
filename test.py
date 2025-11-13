# import os
# import numpy as np
# import tflite_runtime.interpreter as tflite
# from PIL import Image
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # ==========================
# # 1. Paths and Settings
# # ==========================
# MODEL_PATH = "cnn2d_model.tflite"   # your trained model
# TEST_DIR   = "Test"                 # folder with class subfolders
# IMG_SIZE   = (1024, 1024)           # same as training
# class_names = sorted(os.listdir(TEST_DIR))

# # ==========================
# # 2. Load TFLite Model
# # ==========================
# interpreter = tflite.Interpreter(model_path=MODEL_PATH)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # ==========================
# # 3. Preprocessing Function
# # ==========================
# def preprocess_image(img_path):
#     img = Image.open(img_path).convert("L")   # grayscale
#     img = img.resize(IMG_SIZE)                # resize
#     img_array = np.array(img, dtype=np.float32)
#     img_array = (img_array / 127.5) - 1.0     # normalize [-1,1]
#     img_array = np.expand_dims(img_array, axis=-1)   # (h, w, 1)
#     img_array = np.expand_dims(img_array, axis=0)    # (1, h, w, 1)
#     return img_array

# # ==========================
# # 4. Evaluation
# # ==========================
# results = {cls: {"correct": 0, "incorrect": 0} for cls in class_names}
# total_correct, total_count = 0, 0

# for class_name in class_names:
#     class_folder = os.path.join(TEST_DIR, class_name)
#     for img_file in os.listdir(class_folder):
#         img_path = os.path.join(class_folder, img_file)
#         if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
#             continue  # skip non-images

#         # preprocess
#         input_data = preprocess_image(img_path)

#         # inference
#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]['index'])[0]

#         predicted_class_idx = np.argmax(output_data)
#         predicted_class = class_names[predicted_class_idx]

#         # update counts
#         if predicted_class == class_name:
#             results[class_name]["correct"] += 1
#             total_correct += 1
#         else:
#             results[class_name]["incorrect"] += 1
#         total_count += 1

# # ==========================
# # 5. Print Results
# # ==========================
# print("\n--- Per-Class Results ---")
# for cls, stats in results.items():
#     print(f"Class '{cls}': Correct = {stats['correct']}, Incorrect = {stats['incorrect']}")

# accuracy = total_correct / total_count if total_count > 0 else 0
# print("\n--- Overall Accuracy ---")
# print(f"Total Images = {total_count}")
# print(f"Correct Predictions = {total_correct}")
# print(f"Overall Accuracy = {accuracy * 100:.2f}%")



import os
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==========================
# 1. Paths and Settings
# ==========================
MODEL_PATH = "cnn2d_model.tflite"   # your trained model
TEST_DIR   = "Test"                 # folder with class subfolders
IMG_SIZE   = (1024, 1024)           # same as training
class_names = sorted(os.listdir(TEST_DIR))

# ==========================
# 2. Load TFLite Model
# ==========================
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================
# 3. Preprocessing Function
# ==========================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")   # grayscale
    img = img.resize(IMG_SIZE)                # resize
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0     # normalize [-1,1]
    img_array = np.expand_dims(img_array, axis=-1)   # (h, w, 1)
    img_array = np.expand_dims(img_array, axis=0)    # (1, h, w, 1)
    return img_array

# ==========================
# 4. Evaluation
# ==========================
results = {cls: {"correct": 0, "incorrect": 0} for cls in class_names}
total_correct, total_count = 0, 0

# For confusion matrix
true_labels = []
pred_labels = []

for class_name in class_names:
    class_folder = os.path.join(TEST_DIR, class_name)
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)
        if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue  # skip non-images

        # preprocess
        input_data = preprocess_image(img_path)

        # inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_class_idx = np.argmax(output_data)
        predicted_class = class_names[predicted_class_idx]

        # record for confusion matrix
        true_labels.append(class_name)
        pred_labels.append(predicted_class)

        # update counts
        if predicted_class == class_name:
            results[class_name]["correct"] += 1
            total_correct += 1
        else:
            results[class_name]["incorrect"] += 1
        total_count += 1

# ==========================
# 5. Print Results
# ==========================
print("\n--- Per-Class Results ---")
for cls, stats in results.items():
    print(f"Class '{cls}': Correct = {stats['correct']}, Incorrect = {stats['incorrect']}")

accuracy = total_correct / total_count if total_count > 0 else 0
print("\n--- Overall Accuracy ---")
print(f"Total Images = {total_count}")
print(f"Correct Predictions = {total_correct}")
print(f"Overall Accuracy = {accuracy * 100:.2f}%")

# ==========================
# 6. Confusion Matrix
# ==========================
cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix for Test Data")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("\nConfusion matrix saved as 'confusion_matrix.png'")

