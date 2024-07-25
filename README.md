## Drifting aimfully repository

The following package enables experimentation with events generated from pixelwise motion of naturalistic images across an unchanging sensor-background

There are _ basic components to the system

1. **Model** 
2. **Datagenerator**



### Notes

* It is tailored for the **CIFAR100 dataset**. The dataset should be modifiable but some code revisions would be necessary

* It is also tailored for a **pixelwise motion** and would require some creative modifications for moving by part of a pixel

* **Speed** is currently constant but a changing speed can be **easily implemented** by tweaking frames2events_emulator.StatefulEmulator.delta_t and the way it is used in em_frame()

* The **conda environment** used for this project can be recreated from requirements.txt with `conda install --yes --file requirements.txt`