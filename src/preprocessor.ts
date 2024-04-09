import * as tf from '@tensorflow/tfjs';
import * as LibSampleRate from '@alexanderolsen/libsamplerate-js';

import { vggishParams } from './config';

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

function calculateRMS(audioData: Float32Array, hopSize: number) {
    const windowSize = 2048;
    const rms = [];

    for (let i = 0; i < audioData.length; i += hopSize) {
        const window = audioData.slice(i, i + windowSize);
        const sum = window.reduce((acc, val) => acc + (val * val), 0);
        const mean = sum / window.length;
        const rmsValue = Math.sqrt(mean);
        rms.push(rmsValue);
    }

    return rms;
}

function calculateRmsT(rms: number[], sampleRate: number, hopSize: number) {
    const numFrames = rms.length;
    const frameDuration = hopSize / sampleRate;
    const rms_t = new Float32Array(numFrames);

    for (let i = 0; i < numFrames; i++) {
        rms_t[i] = i * frameDuration;
    }

    return rms_t;
}

function findNonsilentIntervals(rms: number[], threshold: number) {
    const nonsilentIntervals = [];
    for (let i = 0; i < rms.length; i++) {
        if (rms[i] > threshold) {
            nonsilentIntervals.push(i);
        }
    }
    return nonsilentIntervals;
}

function findContiguousIntervals(nonsilentIndices: number[]) {
    const contiguousIntervals = [];
    let buffer = [];

    for (let i = 0; i < nonsilentIndices.length - 1; i++) {
        buffer.push(nonsilentIndices[i]);

        // Check if the next index is not contiguous
        if (nonsilentIndices[i] + 1 !== nonsilentIndices[i + 1]) {
            contiguousIntervals.push(buffer);
            buffer = [];
        }
    }

    // Add the last interval if buffer is not empty
    if (buffer.length > 0) {
        contiguousIntervals.push(buffer);
    }

    return contiguousIntervals;
}


function concatenateIntervals(intervals: number[][], audioData: Float32Array, sampleRate: number, rmsT: Float32Array) {
    const start_times = intervals.map(interval => rmsT[interval[0]]);
    const end_times = intervals.map(interval => rmsT[interval[interval.length - 1]]);

    const startSamples = start_times.map(time => Math.floor(time * sampleRate));
    const endSamples = end_times.map(time => Math.ceil(time * sampleRate));

    let numNonsilentSamples = 0;
    for (let i = 0; i < startSamples.length; i++) {
        numNonsilentSamples += endSamples[i] - startSamples[i];
    }

    const nonsilentAudio = new Float32Array(numNonsilentSamples);
    let writeStart = 0;

    intervals.forEach((interval, i) => {
        const startSample = startSamples[i];
        const endSample = endSamples[i];
        const intervalNumSamples = endSample - startSample;
        const intervalAudioData = audioData.subarray(startSample, endSample);
        nonsilentAudio.set(intervalAudioData, writeStart);
        writeStart += intervalNumSamples;
    });

    return nonsilentAudio;
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

export async function removeSilence(audioBuffer: AudioBuffer, sampleRate: number): Promise<AudioBuffer | null> {
    const bufferSampleRate = audioBuffer.sampleRate;

    let audioData = mergeChannels(audioBuffer);
    if (bufferSampleRate !== sampleRate) {
        audioData = await resample(audioData, bufferSampleRate, sampleRate);
    }

    // Define parameters for silence removal
    const thresholdOfSilence = 0.0001;

    // Calculate RMS values
    const hopSize = 512;
    const rms = calculateRMS(audioData, hopSize);
    const rmsT = calculateRmsT(rms, sampleRate, hopSize);

    // Find nonsilent intervals
    const nonsilentIntervals = findNonsilentIntervals(rms, thresholdOfSilence);

    // Discard intervals below minimum duration
    const filteredIntervals = findContiguousIntervals(nonsilentIntervals);

    // Check if there are any nonsilent intervals
    if (filteredIntervals.length === 0) {
        console.error("No nonsilent intervals found.");
        return null; // Return null to indicate failure
    }

    // Create a new audio buffer containing only nonsilent intervals
    const nonsilentAudioData = concatenateIntervals(filteredIntervals, audioData, sampleRate, rmsT);

    if (nonsilentAudioData.length === 0) {
        console.error("Nonsilent audio data is empty.");
        return null; // Return null to indicate failure
    }

    // Create an AudioBuffer from the nonsilent audio data
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const newAudioBuffer = audioCtx.createBuffer(1, nonsilentAudioData.length, sampleRate);
    newAudioBuffer.getChannelData(0).set(nonsilentAudioData);

    return newAudioBuffer;
}

// Converts audio waveform into array of examples for VGGish
export async function waveformToExamples(data: AudioBuffer, sampleRate: number) {
    if (data && data.length) {
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