# yolox-yolov5 
1. 在yolox正添加训练yolov5的数据格式，以及测试代码。
2. 目前是在单gpu模式下调试的代码， 多gpu下还需要修改。 
3. 具体修改： 
   +  在yolox/data/datasets/下添加yolo.py和yolo_classes.py 
   +  在yolox/data/datasets/__init__.py中添加  
       - from .yolo import YOLODataset  
       - from .yolo_classes import YOLO_CLASSES 
   +  在yolox/evaluators/下添加 yolo_evaluator.py 
   +  在yolox/evaluators/__init__.py中添加：
       - from .yolo_evaluator import YOLOEvaluator
   +  训练的配置文件参考yolox_yolo_s.py 
4. 修改代码， 多gpu下分别计算各自卡上的map，未做多gpu同步计算。 
5. 增加多gpu计算map的代码 
