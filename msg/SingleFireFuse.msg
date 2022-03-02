Header header

# image detecting on info
uint16 ir_img_width
uint16 ir_img_height
uint16 rgb_img_width
uint16 rgb_img_height

# target type
uint8 IS_BACKGROUND = 0
uint8 IS_TARGET = 1
uint8 IS_UNKNOWN = 255

uint8 target_type

# focuse mode
uint8 USE_NONE = 0
uint8 USE_IR  = 1
uint8 USE_RGB  = 2

uint8 focuse_mode

# the x and y pos (sub-pixel) of the single fire, opencv style coordinates
float32 ir_img_x
float32 ir_img_y
float32 rgb_img_x
float32 rgb_img_y

# translation(ir->rgb)
float32 scale
float32 offset_x
float32 offset_y
