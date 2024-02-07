import {setWasmPaths} from "@tensorflow/tfjs-backend-wasm"
import * as ort from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';
import * as LibSampleRate from '@alexanderolsen/libsamplerate-js';

ort.env.wasm.numThreads = 1;
ort.env.remoteModels = false;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
setWasmPaths(
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/'
);
await tf.setBackend('wasm');

let vggishModelLoaded: boolean = false;
let featureExtractorPath: string;

async function init() {
    try {
        // Set the number of threads for ONNX Runtime Web WASM backend
        ort.env.wasm.numThreads = 1;

        // Set whether to load remote models for ONNX Runtime Web
        ort.env.remoteModels = false;

        // Set the WASM paths for ONNX Runtime Web
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        // Set the WASM paths for TensorFlow.js
        tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/');

        // Set the TensorFlow.js backend to WASM
        await tf.setBackend('wasm');
    } catch (error) {
        console.error('Error initializing TensorFlow:', error);
    }
}

async function setFeatureExtractor(modelPath: string){
    featureExtractorPath = modelPath;
    vggishModelLoaded = true;
}

export async function process(audioBuffer: AudioBuffer) {
    console.log("process func")
    console.log(audioBuffer)
    const startTime = performance.now();

    while (!vggishModelLoaded) {
        console.log('VGGish model is not loaded yet. Please wait.');
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    const preprocessedStartTime = performance.now();
    const preprocessedData = await preprocess(audioBuffer);
    const preprocessedEndTime = performance.now();

    console.log(`Time taken for preprocess: ${preprocessedEndTime - preprocessedStartTime} milliseconds`);

    console.log(preprocessedData);

    if (preprocessedData) {
        const inferenceStartTime = performance.now();
        const ortOutputsList = await runInferenceParallel(preprocessedData);
        const inferenceEndTime = performance.now();

        console.log(`Time taken for inference in parallel: ${inferenceEndTime - inferenceStartTime} milliseconds`);

        const pprocModelPath = './pproc.onnx';
        const pprocSession = await ort.InferenceSession.create(pprocModelPath);

        const ortOutputsTensor = tf.tensor(ortOutputsList);

        console.log("ortOutputsTensor")
        console.log(ortOutputsTensor)

        const pprocInputArray = Array.from(ortOutputsTensor.dataSync());
        const pprocInputName = pprocSession.inputNames[0];
        const pprocInputTensor = new ort.Tensor('float32', pprocInputArray, ortOutputsTensor.shape);
        const pprocInputs = { [pprocInputName]: pprocInputTensor };
        const pprocOutputs = await pprocSession.run(pprocInputs);

        const pprocOutput = pprocOutputs.output;
        return pprocOutput;
    } else {
        console.log("Error computing Log Mel Spectrogram.");
    }
}

async function runInferenceParallel(inputData: tf.Tensor4D) {
    try {
        const session = await ort.InferenceSession.create(featureExtractorPath);
        
        const [batchSize, channels, height, width] = inputData.shape;

        const promises = [];

        for (let batch = 0; batch < batchSize; batch++) {
            const promise = (async () => {
                const input_data_batch = inputData.slice([batch, 0, 0, 0], [1, 1, height, width]);
                const input_data_onnx = input_data_batch.transpose([0, 2, 1, 3]).reshape([1, 64, 96]);
                const inputArray = Array.from(input_data_onnx.dataSync());

                const inputDims = [1, 64, 96];
                const inputTensor = new ort.Tensor('float32', inputArray, inputDims);

                const feeds = {
                    'melspectrogram': inputTensor,
                };

                const results = await session.run(feeds);
                const outputTensor = results.embeddings;
                const outputArray = [ ...outputTensor.data ];

                return outputArray;
            })();

            promises.push(promise);
        }

        const ortOutputsList = await Promise.all(promises);

        return ortOutputsList;
    } catch (error) {
        console.error('Error during inference:', error);
        return null;
    }
}

async function preprocess(audioBuffer: AudioBuffer) {
    const vggishParams = {
        SAMPLE_RATE: 16000,
        STFT_WINDOW_LENGTH_SECONDS: 0.025,
        STFT_HOP_LENGTH_SECONDS: 0.010,
        NUM_MEL_BINS: 64,
        MEL_MIN_HZ: 125,
        MEL_MAX_HZ: 7500,
        LOG_OFFSET: 0.01,
        EXAMPLE_WINDOW_SECONDS: 0.96,
        EXAMPLE_HOP_SECONDS: 0.96,
    };

    async function waveformToExamples(data: AudioBuffer, sampleRate: number) {
        if (data && data.length) {
            console.log("in waveformToExamples")
            console.log(data)
            console.log(data.getChannelData(0))
            
            // data = (data.numberOfChannels > 1) ? mergeChannels(data) : data;
            console.log("Input data before mergeChannels:", data);
            data = mergeChannels(data)

            if (sampleRate !== vggishParams.SAMPLE_RATE) {
                console.log("Resampling");
                data = await resample(data, sampleRate, vggishParams.SAMPLE_RATE);
                console.log("post resample")
                console.log(data)

                // wavBlob = createWavBlob(data, vggishParams.SAMPLE_RATE);

                // a = document.createElement('a');
                // a.href = URL.createObjectURL(wavBlob);
                // a.download = 'resampled-js.wav';
                // a.click();
            }
            const logMel = await computeLogMelSpectrogram(data, vggishParams.SAMPLE_RATE);
            if (logMel) {      
                const featuresSampleRate = 1.0 / vggishParams.STFT_HOP_LENGTH_SECONDS;
                const exampleWindowLength = Math.round(vggishParams.EXAMPLE_WINDOW_SECONDS * featuresSampleRate);
                const exampleHopLength = Math.round(vggishParams.EXAMPLE_HOP_SECONDS * featuresSampleRate);
                const logMelExamples = frame(logMel, exampleWindowLength, exampleHopLength);

                const logMelTensor = tf.tensor(logMelExamples, undefined, 'float32');

                const expandedTensor = logMelTensor.expandDims(1);
                return expandedTensor;
            } else {
                console.log("Error computing Log Mel Spectrogram.");
            }
        } else {
            console.log("Invalid or undefined data received.");
            return null;
        }
    }


    function createWavBlob(data: Float32Array, sampleRate: number) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
        const blockAlign = numChannels * (bitsPerSample / 8);
        const dataSize = data.length * numChannels * (bitsPerSample / 8);
        const fileSize = 36 + dataSize;

        const buffer = new ArrayBuffer(fileSize);
        const view = new DataView(buffer);

        // RIFF chunk descriptor
        view.setUint32(0, 0x52494646, false); // 'RIFF'
        view.setUint32(4, fileSize - 8, true); // File size - 8

        view.setUint32(8, 0x57415645, false); // 'WAVE'

        // Format chunk
        view.setUint32(12, 0x666d7420, false); // 'fmt '
        view.setUint32(16, 16, true); // Format chunk size
        view.setUint16(20, 1, true); // Audio format (1 for PCM)
        view.setUint16(22, numChannels, true); // Number of channels
        view.setUint32(24, sampleRate, true); // Sample rate
        view.setUint32(28, byteRate, true); // Byte rate
        view.setUint16(32, blockAlign, true); // Block align
        view.setUint16(34, bitsPerSample, true); // Bits per sample

        // Data chunk
        view.setUint32(36, 0x64617461, false); // 'data'
        view.setUint32(40, dataSize, true); // Data chunk size

        // Write audio data
        for (let i = 0; i < data.length; i++) {
            // Assuming 16-bit signed PCM
            const offset = 44 + i * 2;
            if (offset + 2 <= buffer.byteLength) {
                view.setInt16(offset, data[i] * 0x7FFF, true);
            } else {
                console.error("Offset is outside the bounds of the DataView");
                break;
            }
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    function mergeChannels(audioBuffer: AudioBuffer): Float32Array {
        console.log("merging channels")
        const numChannels = audioBuffer.numberOfChannels;
        const numSamples = audioBuffer.length;
        const channels = [];

        for (let i = 0; i < numChannels; i++) {
            console.log(i)
            console.log(audioBuffer.getChannelData(i))
            channels.push(audioBuffer.getChannelData(i));
        }

        const merged = new Float32Array(numSamples).fill(0);

        for (let i = 0; i < numChannels; i++) {
            for (let j = 0; j < numSamples; j++) {
                merged[j] += channels[i][j] / numChannels;
            }
        }

        console.log("Result after mergeChannels:", merged);
        return merged;
    }

    async function resample(data: Float32Array, inputSampleRate: number, outputSampleRate: number) {
        try {
            console.log("in resample")
            console.log(data)
            
            // if (data instanceof AudioBuffer) {
            //     console.log("Converting AudioBuffer to Float32Array");
            //     data = data.getChannelData(0);
            // }

            const src = await LibSampleRate.create(1, inputSampleRate, outputSampleRate, {
                // converterType: LibSampleRate.ConverterType.SRC_SINC_BEST_QUALITY,
                converterType: LibSampleRate.ConverterType.SRC_LINEAR
            });

            const resampledData = await src.full(data);
            // const resampledData = src.simple(data);

            src.destroy();

            console.log("resampledData")
            console.log(resampledData)
            return resampledData;
        } catch (error) {
            console.error("Resample error: ", error);
            throw error;
        }
    }

    async function computeLogMelSpectrogram(data: Float32Array, audioSampleRate: number) {
        try {
            const logOffset = vggishParams.LOG_OFFSET;
            const windowLengthSecs = vggishParams.STFT_WINDOW_LENGTH_SECONDS;
            const hopLengthSecs = vggishParams.STFT_HOP_LENGTH_SECONDS;

            const windowLengthSamples = Math.round(audioSampleRate * windowLengthSecs);
            const hopLengthSamples = Math.round(audioSampleRate * hopLengthSecs);
            const fftLength = Math.pow(2, Math.ceil(Math.log2(windowLengthSamples)));

            const spectrogram = await stftMagnitude(
                data,
                fftLength,
                hopLengthSamples,
                windowLengthSamples
            );

            if (spectrogram) {
                console.log("Spectrogram calculated");
                const melSpectrogram = await computeLogMelFeatures(spectrogram, audioSampleRate);
                if (melSpectrogram) {
                    return melSpectrogram;
                } else {
                    console.log("Mel Spectrogram is undefined or null after computeLogMelFeatures.");
                    return null;
                }
            } else {
                console.log("Spectrogram is undefined or null.");
                return null;
            }
        } catch (error) {
            console.error("Error computing Log Mel Spectrogram: ", error);
            return null;
        }
    }

    async function computeLogMelFeatures(spectrogram: number[][], audioSampleRate: number) {
        try {
            const melSpectrogram = await melSpectrogramFromSpectrogram(
                spectrogram,
                audioSampleRate,
                vggishParams.STFT_WINDOW_LENGTH_SECONDS,
                vggishParams.STFT_HOP_LENGTH_SECONDS,
                vggishParams.NUM_MEL_BINS
            );

            if (melSpectrogram) {
                return applyLogOffset(melSpectrogram, vggishParams.LOG_OFFSET);
            } else {
                console.log("Mel Spectrogram is undefined or null after melSpectrogramFromSpectrogram.");
                return null;
            }
        } catch (error) {
            console.error("Error computing Log Mel Features: ", error);
            return null;
        }
    }

    async function melSpectrogramFromSpectrogram(spectrogram: number[][], audioSampleRate: number, windowLengthSecs: number, hopLengthSecs: number, numMelBins: number) {
        try {
            if (!spectrogram || !spectrogram.length) {
                console.log("Input spectrogram is undefined or has no length.");
                return null;
            }

            const numSpectrogramBins = spectrogram[0].length;

            const melMatrix = await spectrogramToMelMatrix(
                vggishParams.NUM_MEL_BINS,
                numSpectrogramBins,
                audioSampleRate,
                vggishParams.MEL_MIN_HZ,
                vggishParams.MEL_MAX_HZ
            );

            const melSpectrogram = [];

            for (let i = 0; i < spectrogram.length; i++) {
                try {
                    const melSpectrum = applyMelMatrix(spectrogram[i], melMatrix);
                    melSpectrogram.push(melSpectrum);
                } catch (applyMelMatrixError) {
                    console.error("Error in applyMelMatrix:", applyMelMatrixError);
                    throw applyMelMatrixError;
                }
            }
            
            return melSpectrogram;
        } catch (error) {
            console.error("Error in melSpectrogramFromSpectrogram:", error);
            throw error;
        }
    }

    function applyMelMatrix(frameData: number[], melMatrix: number[][]) {
        const melFrame = new Array(melMatrix[0].length).fill(0);

        for (let i = 0; i < melMatrix[0].length; i++) {
            for (let j = 0; j < melMatrix.length; j++) {
                melFrame[i] += frameData[j] * melMatrix[j][i];
            }
        }

        return melFrame;
    }

    function applyLogOffset(melSpectrogram: number[][], logOffset: number) {
        const logMelSpectrogram = melSpectrogram.map(melFrame => {
            const logFrame = melFrame.map(value => Math.log(Math.max(value, logOffset)));
            return logFrame;
        });

        return logMelSpectrogram;
    }

    function linspace(start: number, end: number, numPoints: number) {
        const step = (end - start) / (numPoints - 1);
        return Array.from({ length: numPoints }, (_, i) => start + step * i);
    }

    async function spectrogramToMelMatrix(numMelBins: number, numSpectrogramBins: number, audioSampleRate: number, lowerEdgeHertz: number, upperEdgeHertz: number) {
        try {
            const nyquistHertz = audioSampleRate / 2;
            if (lowerEdgeHertz < 0.0 || lowerEdgeHertz >= upperEdgeHertz || upperEdgeHertz > nyquistHertz) {
                throw new Error("Invalid frequency range for mel spectrogram computation");
            }

            const spectrogramBinsHertz = Array.from({ length: numSpectrogramBins }, (_, i) => (nyquistHertz * i) / (numSpectrogramBins - 1));
            const spectrogramBinsMel = hertzToMel(spectrogramBinsHertz);

            const lowerEdgeMel = hertzToMel(lowerEdgeHertz);
            const upperEdgeMel = hertzToMel(upperEdgeHertz);

            const bandEdgesMel = linspace(lowerEdgeMel[0], upperEdgeMel[0], numMelBins + 2);

            const melWeightsMatrix = Array.from({ length: numSpectrogramBins }, (_, i) => {
                return Array.from({ length: numMelBins }, (_, j) => {
                    const lowerEdgeMel = bandEdgesMel[j];
                    const centerMel = bandEdgesMel[j + 1];
                    const upperEdgeMel = bandEdgesMel[j + 2];
            
                    const lowerSlope = (spectrogramBinsMel[i] - lowerEdgeMel) / (centerMel - lowerEdgeMel);
                    const upperSlope = (upperEdgeMel - spectrogramBinsMel[i]) / (upperEdgeMel - centerMel);
            
                    return Math.max(0.0, Math.min(lowerSlope, upperSlope));
                });
            });

            melWeightsMatrix[0].fill(0);

            return melWeightsMatrix;
        } catch (error) {
            console.error("Error in spectrogramToMelMatrix: ", error);
            throw error;
        }
    }

    function hertzToMel(frequenciesHertz: number[] | number) {
        if (!Array.isArray(frequenciesHertz)) {
            frequenciesHertz = [frequenciesHertz];
        }
        const melBreakFrequencyHertz = 700.0;
        const melHighFrequencyQ = 1127.0;
        const result = frequenciesHertz.map((frequency: number) => melHighFrequencyQ * Math.log(1.0 + frequency / melBreakFrequencyHertz));
        return result;
    }

    async function stftMagnitude(signal: Float32Array, fftLength: number, hopLength: number, windowLength: number) {
        if (!signal || signal.length === 0) {
            console.error("Input signal is undefined or empty.");
            return null;
        }

        const frames = frame(signal, windowLength, hopLength);
        const window = periodicHann(windowLength);

        const windowedFrames = frames.map(frameData => 
            frameData.map((value, index: number) => value * window[index])
        );

        const inputTensor = tf.tensor(windowedFrames, [windowedFrames.length, windowedFrames[0].length], 'float32');

        const complexResult = tf.spectral.rfft(inputTensor, fftLength);

        const magnitudes = tf.abs(complexResult);

        const magnitudesArray = await magnitudes.array();

        return magnitudesArray;
    }

    function frame(data: Float32Array, windowLength: number, hopLength: number) {
        const numSamples = data.length;
        const numFrames = 1 + Math.floor((numSamples - windowLength) / hopLength);

        const frames = [];

        for (let i = 0; i < numFrames; i++) {
            const start = i * hopLength;
            const end = start + windowLength;
            const frameData = data.slice(start, end);
            frames.push(frameData);
        }
        
        return frames;
    }

    function periodicHann(windowLength: number) {
        const window = new Array(windowLength);
        for (let i = 0; i < windowLength; i++) {
            window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / windowLength);
        }
        return window;
    }

    try {
        const spectrogram = await waveformToExamples(audioBuffer, audioBuffer.sampleRate);

        if (spectrogram) {
            return spectrogram;
        } else {
            console.log("Error computing Log Mel Spectrogram.");
        }
    } catch (error) {
        console.error("Error in preprocess:", error);
    }
}

export { setFeatureExtractor };