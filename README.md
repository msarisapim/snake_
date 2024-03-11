# Hand-track snake game with YOLOv8 object detection

## Overview
Enhanced understanding of AI programming fundamentals through the development of a Snake game. Integrated YOLOv8 for hand-tracking controls and utilized Pygame for game development, demonstrating the application of AI in creating interactive gaming experiences.

1. **realtime_handtrack.py**: Train a hand tracking model with the YoloV8 model using a dataset that can be created from webcam/video footage.
2. **extract_vdo2im.py**: Utility script for extracting images from video to create training data.
3. **snake_control.py**: Play the Snake game using mouse control, demonstrating basic game interaction.
4. **main.py**: The main application that combines hand tracking with the Snake game, transitioning control from mouse to hand tracking.
5. **realtime_snake_demo.py**: A backup of main.py for code safety.
   
Dataset: A collection of images for training the hand tracking model, essential for personalizing control interactions.
(You can also download my dataset here (traning set and validation set which include images and labels): https://drive.google.com/drive/folders/1nDy2JfzF4-UTemAdp5ct_WU_chCqEl04?usp=sharing)
