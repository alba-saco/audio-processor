import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import * as tf from '@tensorflow/tfjs';
import { ModelPaths } from './config';

setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/');

export async function initializeTensorFlow(): Promise<void> {
    const backend = 'gpu' in navigator ? 'webgpu' : 'webgl';
    await tf.setBackend(backend);
    await tf.ready();
    console.log(`Using TensorFlow backend: ${backend}`);
}

export async function loadModels(modelPaths: ModelPaths): Promise<{vggishModel: tf.GraphModel, postprocessorModel: tf.GraphModel, voiceModel: tf.GraphModel, reverbModel: tf.GraphModel}> {
    const vggishModelPromise = tf.loadGraphModel(modelPaths.vggishModelPath);
    const postprocessorModelPromise = tf.loadGraphModel(modelPaths.postprocessorModelPath);
    const voiceModelPromise = tf.loadGraphModel(modelPaths.voiceModelPath);
    const reverbModelPromise = tf.loadGraphModel(modelPaths.reverbModelPath);
  
    const [_vggishModel, _postprocessorModel, _voiceModel, _reverbModel] = await Promise.all([
      vggishModelPromise,
      postprocessorModelPromise,
      voiceModelPromise,
      reverbModelPromise
    ]);
  
    return {
        vggishModel: _vggishModel,
        postprocessorModel: _postprocessorModel,
        voiceModel: _voiceModel,
        reverbModel: _reverbModel
      };
}