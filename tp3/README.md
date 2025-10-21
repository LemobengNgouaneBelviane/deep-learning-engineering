TP3 - Convolutional Neural Networks and Computer Vision

This folder contains the practical implementation for TP3 (CNNs & Computer Vision).

Files:
- cnn_classification.py: main script implementing CIFAR-10 data prep, a basic CNN, a small ResNet, and style-transfer extractor prep.
- requirements.txt: python dependencies.
- test_smoke.sh: a lightweight smoke test to run the script.

Run a quick smoke train (recommended inside a virtualenv):

```bash
python3 -m pip install -r requirements.txt
python3 cnn_classification.py --smoke
```

To build and print a small ResNet summary:

```bash
python3 cnn_classification.py --resnet
```