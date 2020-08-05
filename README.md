# Muffler weld classification
This repository contains the code for muffler weld classification of an exhaust pipe.
The system is deployed on an edge device for classifying between a MAG and a laser weld in the welding shop as a poka-yoke to identify and label the welds correctly. 
The real-time system is developed using the pretrained Resnet architecture from the Pytorch library for finetuning the network. The developed low-cost system is robust to lighting and orientation of the component.

![Fig.1: Demonstration of the real-time classification of the welded components](muffler.gif)
