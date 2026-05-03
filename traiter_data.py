import kagglehub
import os

path = kagglehub.dataset_download("akhatova/pcb-defects")
print("Dataset downloaded to:", path)
print(os.listdir(path))

