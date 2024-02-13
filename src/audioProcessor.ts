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

let vggishModelLoaded: boolean = false;
let featureExtractorPath: string;
let classifierPath: string;

function setClassifier(modelPath: string) {
    classifierPath = modelPath;
}

async function setFeatureExtractor(modelPath: string){
    featureExtractorPath = modelPath;
    vggishModelLoaded = true;
}

// Runs feature extractor
async function runFeatureExtractor(audioBuffer: AudioBuffer) {
    console.log("Audio received for processing.")

    if (!vggishModelLoaded) {
        console.log('VGGish model path has not been set. Please use setFeatureExtractor(modelPath)');
        return
    }

    console.log("Running preprocess")
    const preprocessedData = (await preprocess(audioBuffer)) as tf.Tensor4D;

    if (preprocessedData) {
        console.log("Running feature extraction")
        const ortOutputsList = await runInferenceSequence(preprocessedData);

        const pprocModelPath = '../pproc.onnx';
        const pprocSession = await ort.InferenceSession.create(pprocModelPath);

        let ortOutputsTensor;
        if (ortOutputsList) ortOutputsTensor = tf.tensor(ortOutputsList);

        if (ortOutputsTensor !== undefined) {
            const pprocInputArray = Array.from(ortOutputsTensor.dataSync());
            const pprocInputName = pprocSession.inputNames[0];
            const pprocInputTensor = new ort.Tensor('float32', pprocInputArray, ortOutputsTensor.shape);
            const pprocInputs = { [pprocInputName]: pprocInputTensor };
            console.log("Running feature extraction postprocess")
            const pprocOutputs = await pprocSession.run(pprocInputs);

            const pprocOutput = pprocOutputs.output;
            return pprocOutput;
        } else {
            console.log("Error with feature extraction.");
            return null;
        }
    } else {
        console.log("Error in preprocess.");
    }
}

// Runs the classifier model
async function runClassifier(inputData: ort.Tensor) {
    console.log("Running classifier inference")
    const bgSession = await ort.InferenceSession.create(classifierPath);

    const bgInputArray = inputData.data instanceof Float32Array ? Array.from(new Float32Array(inputData.data.buffer)) : [];
    const bgInputName = bgSession.inputNames[0];

    const bgInputTensor = new ort.Tensor('float32', bgInputArray, inputData.dims);
    const bgInputs = { [bgInputName]: bgInputTensor };
    const bgOutput = await bgSession.run(bgInputs);

    if (!bgOutput) {
        throw new Error('Classifier output is null or undefined');
    }

    return bgOutput
}

// Runs VGGish inference in series
async function runInferenceSequence(inputData: tf.Tensor4D): Promise<number[][] | null> {
    try {
        const session = await ort.InferenceSession.create(featureExtractorPath);
        
        const [batchSize, channels, height, width] = inputData.shape;

        const ortOutputsList: number[][] = [];

        for (let batch = 0; batch < batchSize; batch++) {
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
            let outputArray: number[] = [];
            for (let i = 0; i < outputTensor.data.length; i++) {
                outputArray.push(Number(outputTensor.data[i]));
            }

            ortOutputsList.push(outputArray);
        }

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

    // Converts audio waveform into array of examples for VGGish
    async function waveformToExamples(data: AudioBuffer, sampleRate: number) {
        if (data && data.length) {
            // Convert to mono
            const mergedData = mergeChannels(data);

            const audioCtx = new window.AudioContext();
            const dataBuffer = audioCtx.createBuffer(1, mergedData.length, vggishParams.SAMPLE_RATE);
            dataBuffer.getChannelData(0).set(mergedData);
            data = dataBuffer;

            if (sampleRate !== vggishParams.SAMPLE_RATE) {
                // Resample to the rate assumed by VGGish
                const resampledData = await resample(data.getChannelData(0), sampleRate, vggishParams.SAMPLE_RATE);

                const audioCtx = new window.AudioContext();
                const dataBuffer = audioCtx.createBuffer(1, resampledData.length, vggishParams.SAMPLE_RATE);
                dataBuffer.getChannelData(0).set(resampledData);
                data = dataBuffer;
            }
            // Compute log mel spectrogram features
            const logMel = await computeLogMelSpectrogram(data.getChannelData(0), vggishParams.SAMPLE_RATE);

            if (logMel) { 
                // Frame features into examples  
                const featuresSampleRate = 1.0 / vggishParams.STFT_HOP_LENGTH_SECONDS;
                const exampleWindowLength = Math.round(vggishParams.EXAMPLE_WINDOW_SECONDS * featuresSampleRate);
                const exampleHopLength = Math.round(vggishParams.EXAMPLE_HOP_SECONDS * featuresSampleRate);
                const logMelExamples = frame3d(logMel, exampleWindowLength, exampleHopLength);

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

    // Converts to mono. If input is already mono it will remain unchanged
    function mergeChannels(audioBuffer: AudioBuffer): Float32Array {
        const numChannels = audioBuffer.numberOfChannels;
        const numSamples = audioBuffer.length;
        const channels = [];

        for (let i = 0; i < numChannels; i++) {
            channels.push(audioBuffer.getChannelData(i));
        }

        const merged = new Float32Array(numSamples).fill(0);

        for (let i = 0; i < numChannels; i++) {
            for (let j = 0; j < numSamples; j++) {
                merged[j] += channels[i][j] / numChannels;
            }
        }

        return merged;
    }

    async function resample(data: Float32Array, inputSampleRate: number, outputSampleRate: number) {
        try {
            const src = await LibSampleRate.create(1, inputSampleRate, outputSampleRate, {
                converterType: LibSampleRate.ConverterType.SRC_LINEAR
            });

            const resampledData = await src.full(data);

            src.destroy();

            return resampledData;
        } catch (error) {
            console.error("Resample error: ", error);
            throw error;
        }
    }

    // Convert waveform to a log magnitude mel-frequency spectrogram
    async function computeLogMelSpectrogram(data: Float32Array, audioSampleRate: number) {
        try {
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
                const melSpectrogram = await computeLogMelFeatures(spectrogram, audioSampleRate);

                if (melSpectrogram) {
                    return melSpectrogram;
                } else {
                    console.log("Mel Spectrogram is undefined or null.");
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
                    console.error("Error:", applyMelMatrixError);
                    throw applyMelMatrixError;
                }
            }
            
            return melSpectrogram;
        } catch (error) {
            console.error("Error:", error);
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

    // Return a matrix that can post-multiply spectrogram rows to make mel
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
            console.error("Error: ", error);
            throw error;
        }
    }

    // Convert frequencies to mel scale using HTK formula
    function hertzToMel(frequenciesHertz: number[] | number) {
        if (!Array.isArray(frequenciesHertz)) {
            frequenciesHertz = [frequenciesHertz];
        }
        const melBreakFrequencyHertz = 700.0;
        const melHighFrequencyQ = 1127.0;
        const result = frequenciesHertz.map((frequency: number) => melHighFrequencyQ * Math.log(1.0 + frequency / melBreakFrequencyHertz));
        return result;
    }

    // Calculate short-time Fourier transform magnitude
    async function stftMagnitude(signal: Float32Array, fftLength: number, hopLength: number, windowLength: number): Promise<number[][] | null> {
        if (!signal || signal.length === 0) {
            console.error("Input signal is undefined or empty.");
            return null;
        }

        const frames = frame2d(signal, windowLength, hopLength);
        const window = periodicHann(windowLength);

        const windowedFrames = frames.map((frameData: number[]) => 
            frameData.map((value: number, index: number) => value * window[index])
        );

        const inputTensor = tf.tensor(windowedFrames, [windowedFrames.length, windowedFrames[0].length], 'float32');

        const complexResult = tf.spectral.rfft(inputTensor, fftLength);

        const magnitudes = tf.abs(complexResult);

        const magnitudesArray = await magnitudes.array() as number[][];

        return magnitudesArray;
    }

    // Convert 1d array into a 2d sequence of successive possibly overlapping frames
    function frame2d(data: Float32Array, windowLength: number, hopLength: number): number[][] {
        const numSamples = data.length;
        const numFrames = Math.floor((numSamples - windowLength) / hopLength) + 1; 
        const frames: number[][] = [];
            
        for (let i = 0; i < numFrames; i++) {
            const start = i * hopLength;
            const end = start + windowLength;
            const frameData = Array.from(data.slice(start, end));
            frames.push(frameData);
        }
    
        return frames;
    }

    // Convert 2d array into a 3d sequence of successive possibly overlapping frames
    function frame3d(data: number[][], windowLength: number, hopLength: number): number[][][] {
        const numSamples = data.length; 
        const numFrames = Math.floor((numSamples - windowLength) / hopLength) + 1;
        const frames: number[][][] = [];
        
        for (let i = 0; i < numFrames; i++) {
            const start = i * hopLength;
            const end = start + windowLength;
            const frameData = data.slice(start, end)
            frames.push(frameData);
        }
    
        return frames;
    }

    // Calculate a "periodic" Hann window
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
            console.log("Error computing Spectrogram.");
        }
    } catch (error) {
        console.error("Error in preprocess:", error);
    }
}

export { setFeatureExtractor, runFeatureExtractor, setClassifier, runClassifier }
