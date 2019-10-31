## Convolutional Neural Network for animal location tracking

A small library to create and train a convolutional neural network with  thermal imaging from an MLX90640 sensor and video feed before predicting animal location based only on a video stream.


### Hardware Requirements:

Raspberry pi (4b recommended)
MLX90640 thermal sensor
Raspberry pi camera module,
Webcam (optional, logitech c270 (budget) or c920 (better) are great)


### Setup

With Raspbian installed, git clone repo onto local device
Install and compile OpenCV [instructions here](https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/)
 Connect MLX90640 to I2C ports and camera module
Deploy and run camera_recorder.py from rasp_pi_scripts
Should show preview of video feed
Install and compile MLX90640 drivers and test if working
[MLX90640 Python example repo with code this is based on](https://github.com/leswright1977/mlx90640_python)
[Repo with drivers to be compiled](https://github.com/pimoroni/mlx90640-library)


###Example Images

| Predictor | Target | Combination |
| ---- | --- | --- |
| ![rawgif](images/rawvid.gif) | ![heatmap](images/heatmap.gif) | ![combo](images/combo.gif) |