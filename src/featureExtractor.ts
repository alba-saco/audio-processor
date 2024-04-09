import * as tf from '@tensorflow/tfjs';

import { removeSilence, waveformToExamples } from './preprocessor'

async function preprocess(audioBuffer: AudioBuffer) {
    try {
        const noSilenceAudioBuffer = await removeSilence(audioBuffer, 16000)
        const spectrogram = noSilenceAudioBuffer? await waveformToExamples(noSilenceAudioBuffer, noSilenceAudioBuffer.sampleRate) : null;

        if (spectrogram) {
            return spectrogram;
        } else {
            console.log("Error computing Spectrogram.");
        }
    } catch (error) {
        console.error("Error in preprocess:", error);
    }
}

export async function runFeatureExtraction(audioBuffer: AudioBuffer, vggishModel: tf.GraphModel, postprocessorModel: tf.GraphModel) {
    const inputData = await preprocess(audioBuffer);

    try {
      const outputs = [];

      if (!inputData) {
          throw new Error('inputData is undefined');
      }

      const [batchSize, channels, height, width] = inputData.shape;

      const outputPromises = [];
      for (let batch = 0; batch < batchSize; batch++) {
          // Assuming input_data is a 3D tensor (similar to permute(2, 1, 0) in Python)
          const inputDataBatch = inputData.slice([batch, 0, 0, 0], [1, 1, height, width]);

          const inputDataTfjs = inputDataBatch.transpose([0, 2, 1, 3]).reshape([1, 64, 96]);

          // Run inference using the TensorFlow.js model
          outputPromises.push((vggishModel.predict(inputDataTfjs) as tf.Tensor).array());
      }
      const outputArrays = await Promise.all(outputPromises);
      outputs.push(...outputArrays);

      const outputsTensor = tf.tensor(outputs);
      const flattenedOutputsTensor = outputsTensor.reshape([-1, 128]);

      const pprocOutputTensor = postprocessorModel.predict(flattenedOutputsTensor);

      const postProcessedOutput = await (pprocOutputTensor as tf.Tensor).array();
      return postProcessedOutput;  
    } catch (error) {
      console.error('Error during inference:', error);
      return null;
    }
}