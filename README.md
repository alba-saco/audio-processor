# Web Audio Classifier

## Description
The web audio classifier runs a classification model to detect whether a user-submitted audio file is suitable for training a voice model.
The audio is first run through a voice classifier model, which detects whether the submitted audio is singing voice or spoken voice. If the audio is instrumental or is voiced with an instrumental background, it will detect the audio as 'not useable'. If the model detects singing voice, it will then run the reverb classification model to detect whether or not there is reverb applied to the singing voice.

## Installation
```npm install web-audio-processor```

## How to use
### Import

```import { init, runClassifier } from 'web-audio-classifier';```

### Usage
Prior to running the classifier, the model paths need to be set.

[Here](https://console.cloud.google.com/storage/browser/voice-classifier-models) is the link to the TFJS model files. There are 4 models used in this package, each of them are in a subdirectory in the linked storage bucket. Each model has a model.json and shard .bin files.

To initialize the package, the model files need to be passed via the init function. For example:
```javascript
const vggishModelPromise = tf.loadGraphModel('PATH_TO_MODELS_DIR/vggish-tfjs/model.json');
const postprocessorModelPromise = tf.loadGraphModel('PATH_TO_MODELS_DIR/pproc-tfjs/model.json');
const voiceModelPromise = tf.loadGraphModel('PATH_TO_MODELS_DIR/vnv-tfjs/model.json');
const reverbModelPromise = tf.loadGraphModel('PATH_TO_MODELS_DIR/reverb-tfjs/model.json');

Promise.all([vggishModelPromise, postprocessorModelPromise, voiceModelPromise, reverbModelPromise])
    .then(([vggishModel, postprocessorModel, voiceModel, reverbModel]) => {
        init(vggishModel, postprocessorModel, voiceModel, reverbModel);
    })
    .catch(error => {
        console.error("Error loading models:", error);
    });
```
Replacing ```PATH_TO_MODELS_DIR``` with the appropriate path to locate the directory.

Then, to run the classifier, simply run:

```javascript
const classifierMessages = runClassifier(audioBuffer);
```

This will output a variable of type ```string[]```, for example: ```["Detected singing voice", "Detected reverb"]```.