import math
import numpy as np
import cv2

IR_upper_left = [502, 341]
IR_lower_right = [515, 358]

RGB_upper_left = [480, 380]
RGB_lower_right = [491, 388]

IR_diag = math.sqrt((IR_lower_right[0] - IR_upper_left[0])**2 +
                    (IR_lower_right[1] - IR_upper_left[1])**2)

RGB_diag = math.sqrt((RGB_lower_right[0] - RGB_upper_left[0])**2 +
                    (RGB_lower_right[1] - RGB_upper_left[1])**2)

scale = RGB_diag/IR_diag

print(scale)

