## Drifting aimfully repository

The following package enables experimentation with events generated from pixelwise motion of naturalistic images across an unchanging sensor-background

There are _ basic components to the system


1. **Event generator** - see Data/frames2events_emulator.py called from Data/datagenerator.py
2. **Walk** - policy to move image across the unchanging background. see walk/
3. **Model** - processor and determinant of information content in walk. see matrixlstm/classification/net_matrixlstm_vit 


Requires installation of event-based camera emulator v2e. Follow installation instructions at XXX (add link to repository and paper)


### Notes

* It is tailored for the **CIFAR100 dataset**. The dataset should be modifiable but some code revisions would be necessary

* It is also tailored for a **pixelwise motion** and would require some creative modifications for moving by part of a pixel

* **Speed** is currently constant but a changing speed can be **easily implemented** by tweaking frames2events_emulator.StatefulEmulator.delta_t and the way it is used in em_frame()

* The **conda environment** used for this project can be recreated from requirements.txt with `conda install --yes --file requirements.txt`

* matrixlstm_vit is build off of matrixlstm/classification/layers/MatrixLSTM.py. I rewrote their scripts as py files due to compatibility issues but otherwise the fundamental difference is feeding the matrixlstm into a pretrained vit which can be done with a few lines of code (see matrixlstm/classification/models/net_matrixlstm_vit.py which is slightly modified net_matrixlstm_resnet.py). 
    * I used a receptive field of 1  
