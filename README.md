# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.


### Future improvements
* improve inference speed without significant drop in performance by changing the precision of some of the models
* benchmark the running times of different parts of the preprocessing and inference pipeline and let the user specify the CLI argument if they want to see the benchmark timing. Use the get_perf_counts API to print the time it takes for each layer in the model
* use the VTune Amplifier to find hotspots in your Inference Engine Pipeline
* Handle edge cases like multiple people in the video or change in lighting
* add a toggle to the UI to shut off the camera feed and show stats only. How does this affects performance and power?
* Allow the user to select their input option in the command line arguments 