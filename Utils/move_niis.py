import os
import shutil

for root, dirs, files in os.walk("."):
    for name in files:
        if name.endswith((".nii")):
            print(os.path.join(root,name))
            os.rename(os.path.join(root,name), "./"+name)
