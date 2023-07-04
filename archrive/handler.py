import torch
import os, os.path as osp
import logging
import time
import json
import numpy as np

from PIL import Image
import base64
import io


logger = logging.getLogger(__name__)
from sub_module.configs.config import Config

from sub_module.mmdet.inference import build_detector, parse_inference_result
from sub_module.mmdet.modules.dataparallel import build_dp
from sub_module.mmdet.data.transforms.compose import Compose
from sub_module.mmdet.data.dataloader import collate
from sub_module.mmdet.scatter import parallel_scatter

from sub_module.mmdet.get_info_algorithm import Get_info

CONFIG_PATH = "config.py"
request_key = 'data'

class CustomHandler():
    
    def __init__(self):
        self.map_location = None
        self.manifest = None
        self.cfg = None  
        self.torch_device = None       # class 'torch.device'      
        self.initialized = False
        

    def initialize(self, context):        
        self.manifest, self.properties, self.cfg = self._get_properties(context, CONFIG_PATH)

        # set device resource
        if torch.cuda.is_available() and self.properties.gpu_id is not None:
            self.map_location = "cuda"
            device = self.map_location + ":" + str(self.properties.gpu_id)     
        else:
            self.map_location = "cpu"
            device = self.map_location
        if self.cfg.device != device:
            raise ValueError(f"Device setting are not same between config file and properties.\n"
                             f"       cfg.device: {self.cfg.device},     properties: {device}")
        self.device = self.cfg.device

        # Path of .pth file
        model_dir = self.properties.model_dir
        self.model_pth_path = None
        if "serializedFile" in self.manifest.model:
            serialized_file = self.manifest.model.serializedFile
            self.model_pth_path = osp.join(model_dir, serialized_file)
            if not osp.isfile(self.model_pth_path):
                raise RuntimeError(f"Missing the {self.model_pth_path} file")
        else:
            raise ValueError(f"The value `serializedFile` is not defined in manifest")
        
        
        self.model = self._load_model(self.model_pth_path)
        self.model.eval()
        
        self.initialized = True  
    
    def _load_model(self, model_pth_path):
        """ Load the model from the given model path.
        """
        model_format = model_pth_path.split(".")[-1]
        if model_format != "pth":
            raise ValueError(f"The server only for .pth format model, but got {model_format}")

        model = build_detector(self.cfg, model_pth_path, device = self.device, logger = logger)
        model = build_dp(model = model, 
                         cfg = self.cfg,
                         device = self.device,
                         classes = model.CLASSES)
        
        return model
    
    def _get_properties(self, context, config_path):
        """ Load config data to build model and run inference.
        """
        manifest = Config(context.manifest)          # get data from MANIFEST.json in .mar file
        properties = Config(context.system_properties)      # get data from config.properties

        if not osp.isfile(config_path):
            raise RuntimeError("Missing the config.py file")
        cfg = Config.fromfile(config_path)
        return manifest, properties, cfg
        
    
    def postprocess(self, result, classes):
        bboxes, labels, _ = parse_inference_result(result)  # _: mask
        
        get_info_instance = Get_info(bboxes, labels, classes, score_thr = self.cfg.get('show_score_thr', 0.5))
        plate_info_list = get_info_instance.get_board_info()
        if len(plate_info_list) == 0: return None
        return plate_info_list
    
    def inference(self, data, *args, **kwargs):
        if 'img_metas' and 'img' not in data.keys():
            RuntimeError(f"The input data to model must have keys named 'img_meta' and 'img'"
            f"              data.keys() : {data.keys()}")
        with torch.no_grad():
            results = self.model(return_loss=False, rescal = False, **data)        # call model.forward
            
        result = results[0]    
        
        return result
    

    def image_processing(self, img):
        pipeline = Compose(self.cfg.infer_pipeline)
        
        input_dict = dict(file_path = None,
                          file_name = None,
                          img = img,
                          img_shape = img.shape,
                          ori_shape = img.shape,
                          img_fields = ['img'],
                          scale = self.cfg.img_scale)
        data = pipeline(input_dict)
        return data

        
    def preprocess(self, data):
        body = data[0]['body']
        if request_key not in body.keys():
            RuntimeError(f"The body of request not match the input")
        data = body[request_key]
        image = base64.b64decode(data)     # <class 'bytes'>
        
        if not isinstance(image, (bytearray, bytes)):
            RuntimeError(f"The type of decoded data is not `bytearray` or `bytes`"
            f"              type(data): {type(image)}")
        
        pil_image = Image.open(io.BytesIO(image))
        numpy_image = np.array(pil_image)
        data = self.image_processing(numpy_image)
        
        data = collate([data], samples_per_gpu=1)

        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        
        data = parallel_scatter(data, [self.device])[0]
        
        return data
        
        
    def handle(self, data, context):               
        model = self.model
        if not next(model.parameters()).is_cuda:
            RuntimeError(f"modules must be is_cuda, but is not")
            
            
        start_time = time.time()
        
        metrics = context.metrics
        
        data = self.preprocess(data)
        result = self.inference(data)
        plate_info_list = self.postprocess(result, classes = model.CLASSES)

        if plate_info_list is None:
            response = dict(response = 'None')
        else:
            response = dict(response = plate_info_list)
            
        
        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        
        return [response]