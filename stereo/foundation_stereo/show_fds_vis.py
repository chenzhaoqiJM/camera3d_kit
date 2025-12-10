import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(f'{code_dir}/../../thirdparty/FoundationStereo')

disparity_nn = np.load(os.path.join(code_dir, './test_outputs/disp.npy'))

# plt.imshow(disparity_nn)
plt.imshow(disparity_nn, cmap='jet')
plt.colorbar()
plt.show()