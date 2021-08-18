# wash_dish

## Abst
- Input RGBD image and end effector force
- Deep neural network and control theory

## Demo
```
roslaunch jsk_2020_4_carry_dish realpr2_tabletop.launch 
roslaucnh wash_dish data_collection.launch 
roseus euslisp/demo.l
```

## play rosbag 
```
roslaunch wash_dish attention_dish.launch
rosbag play BAGNAME.bag
```
