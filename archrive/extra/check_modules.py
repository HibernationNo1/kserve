
if __name__=="__main__":
    
	import pycocotools
	print(f"pycocotools.__file__: {pycocotools.__file__}")

	import matplotlib
	print(f"matplotlib.__version__: {matplotlib.__version__}")

	import terminaltables
	print(f"terminaltables.__version__: {terminaltables.__version__}")

	import nvgpu
	print(f"nvgpu.__version__: {nvgpu.__file__}")

	import pynvml
	print(f"pynvml.__version__: {pynvml.__version__}")

	import PIL
	print(f"PIL.__version__: {PIL.__version__}")

	import addict
	print(f"addict.__version__: {addict.__version__}")


	import torch
	print(f"torch.__version__: {torch.__version__}")
	cuda_version = torch.version.cuda
	print(f"CUDA Version: {cuda_version}")
 
	print(f"torch.cuda.is_available() ; {torch.cuda.is_available()}")
	
	import cv2
	print(f"cv2.__version__: {cv2.__version__}")
