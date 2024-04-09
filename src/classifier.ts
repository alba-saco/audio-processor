import * as tf from '@tensorflow/tfjs';

function mode(arr: number[]): number {
    const counts: { [key: number]: number } = {}; // Define type for counts object

    arr.forEach(val => {
        counts[val] = (counts[val] || 0) + 1;
    });

    let maxCount = 0;
    let modeValue: number | null = null;
    Object.entries(counts).forEach(([key, count]) => {
        const numKey = parseInt(key);
        if (count > maxCount) {
            maxCount = count;
            modeValue = numKey;
        }
    });

    return modeValue!;
  }

export async function runVoiceModel(input: tf.Tensor, voiceModel: tf.GraphModel): Promise<number> {
    const vnvOut = voiceModel.predict(input);

    const vnvPredictionsTensor = tf.argMax((vnvOut as tf.Tensor), 1);
    const vnvPredictionsArray = await vnvPredictionsTensor.array();

    const vnvDetectedClass = mode(vnvPredictionsArray as number[]);

    return vnvDetectedClass;
}

export async function runReverbModel(input: tf.Tensor, reverbModel: tf.GraphModel): Promise<number> {
    const reverbOut = reverbModel.predict(input);

    const reverbPredictionsTensor = tf.argMax((reverbOut as tf.Tensor), 1);
    const reverbPredictionsArray = await reverbPredictionsTensor.array();
    const reverbDetectedClass = mode(reverbPredictionsArray as number[]);

    return reverbDetectedClass;
}