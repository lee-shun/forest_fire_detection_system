# 外场准备工作

## 物品准备

### 烟

1. 木炭
2. 木屑
3. 打火机
4. 引火油
5. 火盆
6. 水瓶子

### 工具

1. 扎带：固定云冠以及无线网卡
2. 剪刀
3. 记号笔与米尺,确定火焰位置

### 供电

1. 外场电源
2. 电源充电器
3. 转换插头
4. 插排

### 路由器相关

1. 路由器：2个, NetGear+Bell
2. 网线: 2根
3. 12V充电器

### 调试程序相关

1. 显示屏幕
2. 转换插座
3. 键盘鼠标
4. 云冠供电转接头
5. 云冠扩展口
6. HDMI --> micro HDMI
7. USB --> type C (2)
8. USB --> micro usb(DJI-精灵提供)

### 飞行相关

1. M300 装备箱
2. 电池箱
3. 精灵 拍摄视频

## 实验步骤

### thermal threshold experiment

- TODO: 地面流程仿真

**实验目标:**

- 确定火焰的合适位置
- IR locater的threshold
- 获得locater 结果视频
- 获得火源定位位置(gps+distance+gimbal angle)+ 计算的火源位置
- 获得完整视频数据(广角, 变焦,红外)

**准备工作:**

- 准备火烟源
- 使用M300确定火源GPS位置

1. 路由器上电
2. 云冠以及M300 上电
3. 开机,分割屏幕,调整变焦距离, 开始录屏
4. `ssh -X shun@192.168....` 连接云冠计算机

- Workspace 1:
  - `roslaunch dji`
  - `ToggleVehicleVideo open`
  - `Heat IR locater`

- Workspace 2:
  - `GPS recorder`
  - 悬停之后,再使用lisar_ranger_finder 测出距离,读出云台角度. 计算火焰的位置(GPS+Local)

- Workspace 3:
  - `Ranger` arrange files

- 多转几次,先复制出locater的记录, 继续测出多组(3组)距离.

### mission task

- TODO: 地面流程仿真
- TODO: 轨迹设计

**实验目标:**

- 获得M300轨迹
- 获得参考路径
- 获得gimabal control数据
- 获得locater 结果视频
- 获得classifier 结果视频
- 获得segmentator 结果视频
- 获得完整视频数据(广角, 变焦,红外)

**准备工作:**

- 加点火
- 确定一下回折路径的距离, 更改yaml文件 在`config` 文件夹下: 已缩小
- 确定起飞点以及起飞的方向问题: 已仿真.

1. 检查飞行器电量,检查移动电源电量,检查无线连接情况, 确认分布式ros环境
2. 开机,分割屏幕,调整变焦距离,开始录屏
3. `ssh -X shun@192.168...`

- Workspace 1:
  - 云冠`roslaunch dji`
  - 云冠`ToggleVehicleVideo open`
  - 云冠`热成像定位器`
  - 云冠`RGB resnet 分类器`

- Workspace 2:
  - 地面站`HandleImageTransport`
  - 地面站`smoke_segmentator`

- Workspace 3:
  - 云冠`GPS recorder`
  - 云冠`single manager`

