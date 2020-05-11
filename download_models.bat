mkdir models
mkdir logs
mkdir results

    
python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name face-detection-adas-binary-0001 -o "C:\Users\vin_p\Github\EdgeAI-CursorController\models"

python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name landmarks-regression-retail-0009 -o "C:\Users\vin_p\Github\EdgeAI-CursorController\models"

python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name head-pose-estimation-adas-0001 -o "C:\Users\vin_p\Github\EdgeAI-CursorController\models"

python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name gaze-estimation-adas-0002 -o "C:\Users\vin_p\Github\EdgeAI-CursorController\models"