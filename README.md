# Web Audio Classifier

## Info
### Description
The web audio classifier runs a classification model to detect whether a user-submitted audio file is suitable for training a voice model.
The audio is first run through a voice classifier model, which detects whether the submitted audio is singing voice or spoken voice. If the audio is instrumental or is voiced with an instrumental background, it will detect the audio as 'not useable'. If the model detects singing voice, it will then run the reverb classification model to detect whether or not there is reverb applied to the singing voice.

### Performance
It takes an average of 0.9 seconds to process a 10-second audio file on chrome using a GPU.

### Models
There are [4 models](https://console.cloud.google.com/storage/browser/voice-classifier-models) used in this package. The largest is the VGGish feature extractor, which is 275MB. The VGGish-FE postprocessor model is 84KB. The voice/non-voice and the reverb models are 5MB each.

### "Good to Train" Output
The voice quality classifier will identify whether acapella singing or spoken voice is detected. Anything that is not acapella speech or singing will be detected as not useable. If singing voice is detected, it will also detect the presence of reverb in the audio. Outputs from the ```runClassifier``` function that idenitify the audio as 'good to train' will either be ```['Detected singing voice', 'Detected no reverb']``` OR ```['Detected spoken voice']```.


## Installation
```npm install web-audio-processor```

## How to use
### Import

```import { init, runClassifier, ModelPaths } from 'web-audio-classifier';```

### Usage
Prior to running the classifier, the model paths need to be set.

[Here](https://console.cloud.google.com/storage/browser/voice-classifier-models) is the link to the TFJS model files. There are 4 models used in this package, each of them are in a subdirectory in the linked storage bucket. Each model has a model.json and shard .bin files.

To initialize the package, the model files need to be passed via the init function. For example:
```javascript
const modelPaths: ModelPaths = {
    vggishModelPath: 'PATH_TO_MODELS_DIR/vggish-tfjs/model.json',
    postprocessorModelPath: 'PATH_TO_MODELS_DIR/pproc-tfjs/model.json',
    voiceModelPath: 'PATH_TO_MODELS_DIR/vnv-tfjs/model.json',
    reverbModelPath: 'PATH_TO_MODELS_DIR/reverb-tfjs/model.json'
};

await init(modelPaths);
```
Replacing ```PATH_TO_MODELS_DIR``` with the appropriate path to locate the directory.

Then, to run the classifier, simply run:

```javascript
const classifierMessages = runClassifier(audioBuffer);
```

This will output a variable of type ```string[]```, for example: ```["Detected singing voice", "Detected reverb"]```.