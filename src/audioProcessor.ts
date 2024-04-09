declare global {
    interface Window { webkitAudioContext: typeof AudioContext }
  }
  
import {setWasmPaths} from "@tensorflow/tfjs-backend-wasm"
import * as tf from '@tensorflow/tfjs';

import { ModelPaths } from './config';
import { initializeTensorFlow, loadModels } from "./tensorFlowLoader";
import { runFeatureExtraction } from "./featureExtractor";
import { runVoiceModel, runReverbModel } from "./classifier";

setWasmPaths(
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/'
);

let vggishModel: tf.GraphModel;
let postprocessorModel: tf.GraphModel;
let voiceModel: tf.GraphModel;
let reverbModel: tf.GraphModel;

async function init(modelPaths: ModelPaths) {
    await initializeTensorFlow();
  
    ({ vggishModel, postprocessorModel, voiceModel, reverbModel } = await loadModels(modelPaths));
  }

// Runs the classifier model
async function runClassifier(audioBuffer: AudioBuffer) {
    const vggishOut = await runFeatureExtraction(audioBuffer, vggishModel, postprocessorModel)
    const classifiersInputTensor = tf.tensor(vggishOut as number[][]).reshape([-1, 128]);

    const vnvDetectedClass = await runVoiceModel(classifiersInputTensor, voiceModel);

    let voice_result: string = "singing voice";

    if (vnvDetectedClass) {
      if (vnvDetectedClass === 2) {
          voice_result = "spoken voice";
      } else {
          voice_result = "not useable";
      }
    }

    let messages: string[] = [`Detected ${voice_result}`];

    if(vnvDetectedClass === 0) {
      const reverbDetectedClass = await runReverbModel(classifiersInputTensor, reverbModel)
      
      let reverb_result: string = reverbDetectedClass ? "reverb" : "no reverb";
      messages.push(`Detected ${reverb_result}`);
    }

    return messages
}

export {init, runClassifier, ModelPaths}
