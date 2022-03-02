Header header

# image shape
uint16 img_width
uint16 img_height

# reszie shape
uint16 resize_img_width
uint16 resize_img_height

# target type
uint8 IS_BACKGROUND = 0
uint8 IS_HEAT = 1
uint8 IS_UNKNOWN = 255

uint8 target_type

# the x and y pos (sub-pixel) of the single fire, opencv style coordinates
float32 img_x
float32 img_y

