# mobilenet_classifier

## Getting Started

### clone repo
`$ git clone https://github.com/u0251077/mobilenet_classifier.git`

### create venv & install requirement
```
virtualenv venv --python=python3.6
source ./venv/bin/activate
pip install -r requirements.txt
```

### download the datasets
` bash download_dataset.sh apple2orange`

- data architecture
```
└──          
   ├── train.py
   ├── test.py
   ├──datasets
   │   └── apple2orange_train
   │           └── apple
   │                  └── 01.jpg
   │                  └── 02.jpg
   │                  └── xxx.jpg
   │           └── orange
   │                  └── 01.jpg
   │                  └── 02.jpg
   │                  └── xxx.jpg
   │   └── apple2orange_eval
   │           └── apple
   │           └── orange
   └── model
         └── model01.h5
```
### train
`python train.py`

### test 
`python test.py`

