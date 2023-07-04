torch-model-archiver \
	--model-name pipeline \
	--version 0.1 \
	--serialized-file archrive/pipeline.pth \
	--extra-files archrive/extra \
	--handler archrive/handler.py \
	--runtime python \
	--export-path gs/model-store