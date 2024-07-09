# CFA: Consistent feature attribution for black-box adversarial attack against object detection



Mingyang Li, Wei Zhang, Yadong Wang, Te Guo, Shilong Wu, Yutong Wang*, Tianyu Shen, Kunfeng Wang*

(*Corresponding authors)



An official implementation of the CFA adversarial patch generation framework.
## Framework Overview

We interact with the middle and deep feature maps of the backbone network of the object detector using the clean image and the corresponding gradients to explore the consistency of the model’s inference logic reflected in the feature maps. We design a consistent feature attribution loss function to guide the adversarial patch to disrupt features beneficial for correct model prediction.

![pipeline.png](pipeline.png)




## Install
### Environment

```bash
conda create -n cfa python=3.7
conda activate cfa
pip install -r requirements.txt
```
Please refer to [PyTorch Docs](https://pytorch.org/get-started/previous-versions/) to install `torch` and `torchvision` for better compatibility.

### Dataset: 
INRIA Person




## Run



#### Evaluation

The evaluation metrics of the **Mean Average Precision([mAP](https://github.com/Cartucho/mAP))** is provided.



```bash
# To run the full command in the root proj dir:
python evaluate.py \
-p ./results/CFA_1000.png \
-cfg ./configs/eval/coco80.yaml \
-lp ./data/INRIAPerson/Test/labels \
-dr ./data/INRIAPerson/Test/pos \
-s ./data/test \
-e 0 # attack class id

# for torch-models(coco91): replace -cfg with ./configs/eval/coco91.yaml

# For detailed supports of the arguments:
python evaluate.py -h
```

#### Training


```bash
# run the full command:
python train_optim_cfa.py -np \
-cfg=advpatch/cfa.yaml \
-s=./results/cfa_test \
-n=cfa_test # patch name & tensorboard name

# For detailed supports of the arguments:
python train_optim.py -h
```
The default save path of tensorboard logs is **runs/**.




## Acknowledgements

* AdvPatch
* T-SEA



## Contact Us
If you have any problem about this work, please feel free to reach us out at `2022200754@buct.edu.cn`.

