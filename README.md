# Forest Fire Detection System

## <p align="center">![flight test 1](./document/flight_test_res/flight_test_1.gif)</p>

**In this work, we implemented an online early forest fire detection system based on drone platform.**

- Multiple types of sensors are employed to detect multiple features of the fire flame and smoke.

- Both deep convolutional neural network (CNN) and traditional computer vision algorithms are implemented to process the
  visual images (RGB images) and infrared images (thermal images).

- Other additional algorithms are also implemented to finish the forest fire detection and geolocation task.

## System architecture

### <div align=center>![system architecture](./document/flight_test_res/system.png)</div>

## Features

| functions                        | results                                              |
|----------------------------------|------------------------------------------------------|
| path planning                    | ![path](./document/flight_test_res/trajectory_1.png) |
| forest fire image classification | ![path](./document/flight_test_res/class.png)        |
| forest fire image segmentation   | ![path](./document/flight_test_res/mask.png)         |
| gimbal control                   | ![path](./document/flight_test_res/gimbal.png)       |
| fire point geolocation           | ![path](./document/flight_test_res/locate.png)       |

## Hardware Platform

- DJI M300 RTK
- H20T Camera
- Nvidia NX on board computer

## Usage

1. install [DJI OnboardSDK](https://developer.dji.com/onboard-api-reference/index.html)
2. install [ROS](https://www.ros.org/), only `melodic` version is well tested and recommneded.
3. `mkdir -p ~/catkin_ws/src`
4. `cd ~/catkin_ws/src/ && catkin_init_workspace`
5. `git clone https://github.com/lee-shun/forest_fire_detection_system.git`
6. `git clone https://github.com/lee-shun/dji_osdk_ros_cv4`
7. `cd ~/catkin_ws/ && catkin build`

## Outdoor flighht test videos

[Flight test 1 on Youtube](https://www.youtube.com/watch?v=dQG73LW8jxQ)

## Copyright

**Copyright (C) 2021 Concordia NAVlab. All rights reserved.**

