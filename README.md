# Web Background Noise Classifier

## How it works
This package is intended to be used with any model that takes the output of the VGGish feature extractor as an input. It includes methods to use to run the feature extraction in browser before passing it through to your model.

## Installation
```npm install web-audio-processor```

## How to use
### Import

```import { setFeatureExtractor, runFeatureExtractor, setClassifier, runClassifier } from 'web-audio-classifier';```

### Usage
Prior to preprocessing, the model paths need to be set. To set the feature extractor path, use ```setFeatureExtractor(vggishModelPath)```. [Here](https://essentia.upf.edu/models/feature-extractors/vggish/) is the link to download the the VGGish onnx model (.onnx extension). To set the classifier model path, use ```setClassifier(classifierModelPath)```.

To run the VGGish feature extraction ```runFeatureExtractor(audioBuffer)```. The output of this can be passed into the classifier.

To run the classifier, use ```runClassifier(inputData)``` where inputData is the output from ```runFeatureExtractor```.
