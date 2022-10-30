import os
import glob
import random
import shutil
from PIL import Image
import numpy as np

if __name__ == '__main__':
    train_files = glob.glob(os.path.join('train','*','*.jpg'))
    print(f'Totally {len(train_files)} files for training')

    result = []
    for file in train_files:
        img = Image.open(file).convert('RGB')
        img = np.array(img).astype(np.uint8)
        img = img/255.
        result.append(img)

    print(np.shape(result))
    mean = np.mean(result,axis = (0, 1, 2))
    std = np.std(result, axis = (0, 1, 2))

    print(f'mean: {mean}  \n std: {std}')