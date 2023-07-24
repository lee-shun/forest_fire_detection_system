# How to run the precisely water dropping node?

1. Open a terminal, start the dji vehicle node.

   ```bash
   roslaunch dji_osdk_ros dji_vehicle_node.launch
   ```

2. Open another terminal, open the H20T camera video.

   ```bash
   rosrun forest_fire_detection_system ToggleVehicleVideo open
   ```

3. Open another terminal, start the thermal camera detecting node.

   ```bash
    rosrun forest_fire_detection_system heat_ir_threshold_locater.py
   ```

4. Open another terminal, run the dropper.

   ```bash
    rosrun forest_fire_detection_system PreciselyWaterDropping
   ```
