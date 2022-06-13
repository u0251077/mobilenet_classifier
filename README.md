# mobilenet_classifier

## Getting Started

### clone repo
`$ git clone https://github.com/u0251077/mobilenet_classifier.git`

### create venv & install requirement
- use poetry
```
poetry env use 3.6
poetry install
poetry run python xxx.py
```

- use virtualenv
```
virtualenv venv --python=python3.6
source ./venv/bin/activate
pip install -r requirement.txt
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

### speed time
- device : rtx 2070 

1. network:mobilenet+fc1024
   - speed time: 0.009s / FPS:111

2. network:mobilenet+fc2048+fc1024+fc512
   - speed time: 0.01s / FPS:100




