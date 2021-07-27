# yolox-yolov5 
1. 在yolox正添加训练yolov5的数据格式，以及测试代码。
2. 目前是在单gpu模式下调试的代码， 多gpu下还需要修改。 
3. 具体修改： 
   +*  在yolox/data/datasets/下添加yolo.py和yolo_classes.py 
   +*  在yolox/data/datasets/__init__.py中添加 
       (1)from .yolo import YOLODataset 
       (2)from .yolo_classes import YOLO_CLASSES 
   +*  在yolox/data/evaluators/下添加 yolo_evaluator.py 
   +*  训练的配置文件参考yolox_yolo_s.py 
