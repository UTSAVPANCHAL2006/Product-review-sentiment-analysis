import os
print("🚀 Starting full training + evaluation pipeline...\n")

os.system("python3 src/data_split.py")
os.system("python3 src/preproces.py")
os.system("python3 src/model.py")
os.system("python3 src/train.py")

print("\n✅ Pipeline completed successfully.")