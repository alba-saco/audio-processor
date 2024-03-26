declare global {
    interface Window { webkitAudioContext: typeof AudioContext }
  }
  
import {setWasmPaths} from "@tensorflow/tfjs-backend-wasm"
import * as tf from '@tensorflow/tfjs';
import * as LibSampleRate from '@alexanderolsen/libsamplerate-js';

import vggishModelJson from "../models/vggish-tfjs/model.json" 

const vggishModelWeight1 = require("../models/vggish-tfjs/group1-shard1of69.bin");
const vggishModelWeight2 = require("../models/vggish-tfjs/group1-shard2of69.bin");
const vggishModelWeight3 = require("../models/vggish-tfjs/group1-shard3of69.bin");
const vggishModelWeight4 = require("../models/vggish-tfjs/group1-shard4of69.bin");
const vggishModelWeight5 = require("../models/vggish-tfjs/group1-shard5of69.bin");
const vggishModelWeight6 = require("../models/vggish-tfjs/group1-shard6of69.bin");
const vggishModelWeight7 = require("../models/vggish-tfjs/group1-shard7of69.bin");
const vggishModelWeight8 = require("../models/vggish-tfjs/group1-shard8of69.bin");
const vggishModelWeight9 = require("../models/vggish-tfjs/group1-shard9of69.bin");
const vggishModelWeight10 = require("../models/vggish-tfjs/group1-shard10of69.bin");
const vggishModelWeight11 = require("../models/vggish-tfjs/group1-shard11of69.bin");
const vggishModelWeight12 = require("../models/vggish-tfjs/group1-shard12of69.bin");
const vggishModelWeight13 = require("../models/vggish-tfjs/group1-shard13of69.bin");
const vggishModelWeight14 = require("../models/vggish-tfjs/group1-shard14of69.bin");
const vggishModelWeight15 = require("../models/vggish-tfjs/group1-shard15of69.bin");
const vggishModelWeight16 = require("../models/vggish-tfjs/group1-shard16of69.bin");
const vggishModelWeight17 = require("../models/vggish-tfjs/group1-shard17of69.bin");
const vggishModelWeight18 = require("../models/vggish-tfjs/group1-shard18of69.bin");
const vggishModelWeight19 = require("../models/vggish-tfjs/group1-shard19of69.bin");
const vggishModelWeight20 = require("../models/vggish-tfjs/group1-shard20of69.bin");
const vggishModelWeight21 = require("../models/vggish-tfjs/group1-shard21of69.bin");
const vggishModelWeight22 = require("../models/vggish-tfjs/group1-shard22of69.bin");
const vggishModelWeight23 = require("../models/vggish-tfjs/group1-shard23of69.bin");
const vggishModelWeight24= require("../models/vggish-tfjs/group1-shard24of69.bin");
const vggishModelWeight25 = require("../models/vggish-tfjs/group1-shard25of69.bin");
const vggishModelWeight26 = require("../models/vggish-tfjs/group1-shard26of69.bin");
const vggishModelWeight27 = require("../models/vggish-tfjs/group1-shard27of69.bin");
const vggishModelWeight28 = require("../models/vggish-tfjs/group1-shard28of69.bin");
const vggishModelWeight29 = require("../models/vggish-tfjs/group1-shard29of69.bin");
const vggishModelWeight30 = require("../models/vggish-tfjs/group1-shard30of69.bin");
const vggishModelWeight31 = require("../models/vggish-tfjs/group1-shard31of69.bin");
const vggishModelWeight32 = require("../models/vggish-tfjs/group1-shard32of69.bin");
const vggishModelWeight33 = require("../models/vggish-tfjs/group1-shard33of69.bin");
const vggishModelWeight34 = require("../models/vggish-tfjs/group1-shard34of69.bin");
const vggishModelWeight35 = require("../models/vggish-tfjs/group1-shard35of69.bin");
const vggishModelWeight36 = require("../models/vggish-tfjs/group1-shard36of69.bin");
const vggishModelWeight37 = require("../models/vggish-tfjs/group1-shard37of69.bin");
const vggishModelWeight38 = require("../models/vggish-tfjs/group1-shard38of69.bin");
const vggishModelWeight39 = require("../models/vggish-tfjs/group1-shard39of69.bin");
const vggishModelWeight40 = require("../models/vggish-tfjs/group1-shard40of69.bin");
const vggishModelWeight41 = require("../models/vggish-tfjs/group1-shard41of69.bin");
const vggishModelWeight42 = require("../models/vggish-tfjs/group1-shard42of69.bin");
const vggishModelWeight43 = require("../models/vggish-tfjs/group1-shard43of69.bin");
const vggishModelWeight44 = require("../models/vggish-tfjs/group1-shard44of69.bin");
const vggishModelWeight45 = require("../models/vggish-tfjs/group1-shard45of69.bin");
const vggishModelWeight46 = require("../models/vggish-tfjs/group1-shard46of69.bin");
const vggishModelWeight47 = require("../models/vggish-tfjs/group1-shard47of69.bin");
const vggishModelWeight48 = require("../models/vggish-tfjs/group1-shard48of69.bin");
const vggishModelWeight49 = require("../models/vggish-tfjs/group1-shard49of69.bin");
const vggishModelWeight50 = require("../models/vggish-tfjs/group1-shard50of69.bin");
const vggishModelWeight51 = require("../models/vggish-tfjs/group1-shard51of69.bin");
const vggishModelWeight52 = require("../models/vggish-tfjs/group1-shard52of69.bin");
const vggishModelWeight53 = require("../models/vggish-tfjs/group1-shard53of69.bin");
const vggishModelWeight54 = require("../models/vggish-tfjs/group1-shard54of69.bin");
const vggishModelWeight55 = require("../models/vggish-tfjs/group1-shard55of69.bin");
const vggishModelWeight56 = require("../models/vggish-tfjs/group1-shard56of69.bin");
const vggishModelWeight57 = require("../models/vggish-tfjs/group1-shard57of69.bin");
const vggishModelWeight58 = require("../models/vggish-tfjs/group1-shard58of69.bin");
const vggishModelWeight59 = require("../models/vggish-tfjs/group1-shard59of69.bin");
const vggishModelWeight60 = require("../models/vggish-tfjs/group1-shard60of69.bin");
const vggishModelWeight61 = require("../models/vggish-tfjs/group1-shard61of69.bin");
const vggishModelWeight62 = require("../models/vggish-tfjs/group1-shard62of69.bin");
const vggishModelWeight63 = require("../models/vggish-tfjs/group1-shard63of69.bin");
const vggishModelWeight64 = require("../models/vggish-tfjs/group1-shard64of69.bin");
const vggishModelWeight65 = require("../models/vggish-tfjs/group1-shard65of69.bin");
const vggishModelWeight66 = require("../models/vggish-tfjs/group1-shard66of69.bin");
const vggishModelWeight67 = require("../models/vggish-tfjs/group1-shard67of69.bin");
const vggishModelWeight68 = require("../models/vggish-tfjs/group1-shard68of69.bin");
const vggishModelWeight69 = require("../models/vggish-tfjs/group1-shard69of69.bin");

setWasmPaths(
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/'
);

let postprocessorModel: tf.GraphModel;
let voiceModel: tf.GraphModel;
let reverbModel: tf.GraphModel;

export class VggishModel {
    model: tf.GraphModel | null = null;

    init = async () => {
        const vggishModelJsonString = JSON.stringify(vggishModelJson);
        const vggishModelJsonBlob = new Blob([vggishModelJsonString], { type: "application/json" });
        const vggishModelJson2 = new File([vggishModelJsonBlob], "vggishModel.json")

        const b1 = Buffer.from(vggishModelWeight1.split(',')[1], 'base64')
        const modelWeights1 = new File([b1], "group1-shard1of69.bin")
        const b2 = Buffer.from(vggishModelWeight2.split(',')[1], 'base64')
        const modelWeights2 = new File([b2], "group1-shard2of69.bin") 
        const b3 = Buffer.from(vggishModelWeight3.split(',')[1], 'base64')
        const modelWeights3 = new File([b3], "group1-shard3of69.bin") 
        const b4 = Buffer.from(vggishModelWeight4.split(',')[1], 'base64')
        const modelWeights4 = new File([b4], "group1-shard4of69.bin") 
        const b5 = Buffer.from(vggishModelWeight5.split(',')[1], 'base64')
        const modelWeights5 = new File([b5], "group1-shard5of69.bin") 
        const b6 = Buffer.from(vggishModelWeight6.split(',')[1], 'base64')
        const modelWeights6 = new File([b6], "group1-shard6of69.bin") 
        const b7 = Buffer.from(vggishModelWeight7.split(',')[1], 'base64')
        const modelWeights7 = new File([b7], "group1-shard7of69.bin") 
        const b8 = Buffer.from(vggishModelWeight8.split(',')[1], 'base64')
        const modelWeights8 = new File([b8], "group1-shard8of69.bin") 
        const b9 = Buffer.from(vggishModelWeight9.split(',')[1], 'base64')
        const modelWeights9 = new File([b9], "group1-shard9of69.bin") 
        const b10 = Buffer.from(vggishModelWeight10.split(',')[1], 'base64')
        const modelWeights10 = new File([b10], "group1-shard10of69.bin") 

        const b11 = Buffer.from(vggishModelWeight11.split(',')[1], 'base64')
        const modelWeights11 = new File([b11], "group1-shard11of69.bin") 
        const b12 = Buffer.from(vggishModelWeight12.split(',')[1], 'base64')
        const modelWeights12 = new File([b12], "group1-shard12of69.bin") 
        const b13 = Buffer.from(vggishModelWeight13.split(',')[1], 'base64')
        const modelWeights13 = new File([b13], "group1-shard13of69.bin") 
        const b14 = Buffer.from(vggishModelWeight14.split(',')[1], 'base64')
        const modelWeights14 = new File([b14], "group1-shard14of69.bin") 
        const b15 = Buffer.from(vggishModelWeight15.split(',')[1], 'base64')
        const modelWeights15 = new File([b15], "group1-shard15of69.bin") 
        const b16 = Buffer.from(vggishModelWeight16.split(',')[1], 'base64')
        const modelWeights16 = new File([b16], "group1-shard16of69.bin") 
        const b17 = Buffer.from(vggishModelWeight17.split(',')[1], 'base64')
        const modelWeights17 = new File([b17], "group1-shard17of69.bin") 
        const b18 = Buffer.from(vggishModelWeight18.split(',')[1], 'base64')
        const modelWeights18 = new File([b18], "group1-shard18of69.bin") 
        const b19 = Buffer.from(vggishModelWeight19.split(',')[1], 'base64')
        const modelWeights19 = new File([b19], "group1-shard19of69.bin") 
        const b20 = Buffer.from(vggishModelWeight20.split(',')[1], 'base64')
        const modelWeights20 = new File([b20], "group1-shard20of69.bin") 

        const b21 = Buffer.from(vggishModelWeight21.split(',')[1], 'base64')
        const modelWeights21 = new File([b21], "group1-shard21of69.bin") 
        const b22 = Buffer.from(vggishModelWeight22.split(',')[1], 'base64')
        const modelWeights22 = new File([b22], "group1-shard22of69.bin") 
        const b23 = Buffer.from(vggishModelWeight23.split(',')[1], 'base64')
        const modelWeights23 = new File([b23], "group1-shard23of69.bin") 
        const b24 = Buffer.from(vggishModelWeight24.split(',')[1], 'base64')
        const modelWeights24 = new File([b24], "group1-shard24of69.bin") 
        const b25 = Buffer.from(vggishModelWeight25.split(',')[1], 'base64')
        const modelWeights25 = new File([b25], "group1-shard25of69.bin") 
        const b26 = Buffer.from(vggishModelWeight26.split(',')[1], 'base64')
        const modelWeights26 = new File([b26], "group1-shard26of69.bin") 
        const b27 = Buffer.from(vggishModelWeight27.split(',')[1], 'base64')
        const modelWeights27 = new File([b27], "group1-shard27of69.bin") 
        const b28 = Buffer.from(vggishModelWeight28.split(',')[1], 'base64')
        const modelWeights28 = new File([b28], "group1-shard28of69.bin") 
        const b29 = Buffer.from(vggishModelWeight29.split(',')[1], 'base64')
        const modelWeights29 = new File([b29], "group1-shard29of69.bin") 
        const b30 = Buffer.from(vggishModelWeight30.split(',')[1], 'base64')
        const modelWeights30 = new File([b30], "group1-shard30of69.bin") 

        const b31 = Buffer.from(vggishModelWeight31.split(',')[1], 'base64')
        const modelWeights31 = new File([b31], "group1-shard31of69.bin") 
        const b32 = Buffer.from(vggishModelWeight32.split(',')[1], 'base64')
        const modelWeights32 = new File([b32], "group1-shard32of69.bin") 
        const b33 = Buffer.from(vggishModelWeight33.split(',')[1], 'base64')
        const modelWeights33 = new File([b33], "group1-shard33of69.bin") 
        const b34 = Buffer.from(vggishModelWeight34.split(',')[1], 'base64')
        const modelWeights34 = new File([b34], "group1-shard34of69.bin") 
        const b35 = Buffer.from(vggishModelWeight35.split(',')[1], 'base64')
        const modelWeights35 = new File([b35], "group1-shard35of69.bin") 
        const b36 = Buffer.from(vggishModelWeight36.split(',')[1], 'base64')
        const modelWeights36 = new File([b36], "group1-shard36of69.bin") 
        const b37 = Buffer.from(vggishModelWeight37.split(',')[1], 'base64')
        const modelWeights37 = new File([b37], "group1-shard37of69.bin") 
        const b38 = Buffer.from(vggishModelWeight38.split(',')[1], 'base64')
        const modelWeights38 = new File([b38], "group1-shard38of69.bin") 
        const b39 = Buffer.from(vggishModelWeight39.split(',')[1], 'base64')
        const modelWeights39 = new File([b39], "group1-shard39of69.bin") 
        const b40 = Buffer.from(vggishModelWeight40.split(',')[1], 'base64')
        const modelWeights40 = new File([b40], "group1-shard40of69.bin") 

        const b41 = Buffer.from(vggishModelWeight41.split(',')[1], 'base64')
        const modelWeights41 = new File([b41], "group1-shard41of69.bin") 
        const b42 = Buffer.from(vggishModelWeight42.split(',')[1], 'base64')
        const modelWeights42 = new File([b42], "group1-shard42of69.bin") 
        const b43 = Buffer.from(vggishModelWeight43.split(',')[1], 'base64')
        const modelWeights43 = new File([b43], "group1-shard43of69.bin") 
        const b44 = Buffer.from(vggishModelWeight44.split(',')[1], 'base64')
        const modelWeights44 = new File([b44], "group1-shard44of69.bin") 
        const b45 = Buffer.from(vggishModelWeight45.split(',')[1], 'base64')
        const modelWeights45 = new File([b45], "group1-shard45of69.bin") 
        const b46 = Buffer.from(vggishModelWeight46.split(',')[1], 'base64')
        const modelWeights46 = new File([b46], "group1-shard46of69.bin") 
        const b47 = Buffer.from(vggishModelWeight47.split(',')[1], 'base64')
        const modelWeights47 = new File([b47], "group1-shard47of69.bin") 
        const b48 = Buffer.from(vggishModelWeight48.split(',')[1], 'base64')
        const modelWeights48 = new File([b48], "group1-shard48of69.bin") 
        const b49 = Buffer.from(vggishModelWeight49.split(',')[1], 'base64')
        const modelWeights49 = new File([b49], "group1-shard49of69.bin") 
        const b50 = Buffer.from(vggishModelWeight50.split(',')[1], 'base64')
        const modelWeights50 = new File([b50], "group1-shard50of69.bin") 
        
        const b51 = Buffer.from(vggishModelWeight51.split(',')[1], 'base64')
        const modelWeights51 = new File([b51], "group1-shard51of69.bin") 
        const b52 = Buffer.from(vggishModelWeight52.split(',')[1], 'base64')
        const modelWeights52 = new File([b52], "group1-shard52of69.bin") 
        const b53 = Buffer.from(vggishModelWeight53.split(',')[1], 'base64')
        const modelWeights53 = new File([b53], "group1-shard53of69.bin") 
        const b54 = Buffer.from(vggishModelWeight54.split(',')[1], 'base64')
        const modelWeights54 = new File([b54], "group1-shard54of69.bin") 
        const b55 = Buffer.from(vggishModelWeight55.split(',')[1], 'base64')
        const modelWeights55 = new File([b55], "group1-shard55of69.bin") 
        const b56 = Buffer.from(vggishModelWeight56.split(',')[1], 'base64')
        const modelWeights56 = new File([b56], "group1-shard56of69.bin") 
        const b57 = Buffer.from(vggishModelWeight57.split(',')[1], 'base64')
        const modelWeights57 = new File([b57], "group1-shard57of69.bin") 
        const b58 = Buffer.from(vggishModelWeight58.split(',')[1], 'base64')
        const modelWeights58 = new File([b1], "group1-shard58of69.bin") 
        const b59 = Buffer.from(vggishModelWeight59.split(',')[1], 'base64')
        const modelWeights59 = new File([b59], "group1-shard59of69.bin") 
        const b60 = Buffer.from(vggishModelWeight60.split(',')[1], 'base64')
        const modelWeights60 = new File([b60], "group1-shard60of69.bin") 

        const b61 = Buffer.from(vggishModelWeight61.split(',')[1], 'base64')
        const modelWeights61 = new File([b61], "group1-shard61of69.bin") 
        const b62 = Buffer.from(vggishModelWeight62.split(',')[1], 'base64')
        const modelWeights62 = new File([b62], "group1-shard62of69.bin") 
        const b63 = Buffer.from(vggishModelWeight63.split(',')[1], 'base64')
        const modelWeights63 = new File([b63], "group1-shard63of69.bin") 
        const b64 = Buffer.from(vggishModelWeight64.split(',')[1], 'base64')
        const modelWeights64 = new File([b64], "group1-shard64of69.bin") 
        const b65 = Buffer.from(vggishModelWeight65.split(',')[1], 'base64')
        const modelWeights65 = new File([b65], "group1-shard65of69.bin") 
        const b66 = Buffer.from(vggishModelWeight66.split(',')[1], 'base64')
        const modelWeights66 = new File([b66], "group1-shard66of69.bin") 
        const b67 = Buffer.from(vggishModelWeight67.split(',')[1], 'base64')
        const modelWeights67 = new File([b67], "group1-shard67of69.bin") 
        const b68 = Buffer.from(vggishModelWeight68.split(',')[1], 'base64')
        const modelWeights68 = new File([b68], "group1-shard68of69.bin") 
        const b69 = Buffer.from(vggishModelWeight69.split(',')[1], 'base64')
        const modelWeights69 = new File([b69], "group1-shard69of69.bin") 

        this.model = await tf.loadGraphModel(tf.io.browserFiles([vggishModelJson2, modelWeights1, modelWeights2, modelWeights3, modelWeights4, modelWeights5, modelWeights6, modelWeights7, modelWeights8, modelWeights9, modelWeights10, modelWeights11, modelWeights12, modelWeights13, modelWeights14, modelWeights15, modelWeights16, modelWeights17, modelWeights18, modelWeights19, modelWeights20, modelWeights21, modelWeights22, modelWeights23, modelWeights24, modelWeights25, modelWeights26, modelWeights27, modelWeights28, modelWeights29, modelWeights30, modelWeights31, modelWeights32, modelWeights33, modelWeights34, modelWeights35, modelWeights36, modelWeights37, modelWeights38, modelWeights39, modelWeights40, modelWeights41, modelWeights42, modelWeights43, modelWeights44, modelWeights45, modelWeights46, modelWeights47, modelWeights48, modelWeights49, modelWeights50, modelWeights51, modelWeights52, modelWeights53, modelWeights54, modelWeights55, modelWeights56, modelWeights57, modelWeights58, modelWeights59, modelWeights60, modelWeights61, modelWeights62, modelWeights63, modelWeights64, modelWeights65, modelWeights66, modelWeights67, modelWeights68, modelWeights69]))
    }

    predict = async (inputData: tf.Tensor<tf.Rank>): Promise<tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[]> =>{
        const output = await this.model?.execute(inputData);
        return output ?? [];
    }
}

let vggishModel: VggishModel;


async function init(){
    initializeTensorFlow();

    // vggishModel = await tf.loadGraphModel('/models/vggish-tfjs/model.json');
    vggishModel = new VggishModel();
    postprocessorModel = await tf.loadGraphModel('/models/pproc-tfjs/model.json');
    voiceModel = await tf.loadGraphModel('/models/vnv-tfjs/model.json');
    reverbModel = await tf.loadGraphModel('/models/reverb-tfjs/model.json');

    console.log("Models loaded")
}

async function initializeTensorFlow(): Promise<void> {
    const backend = 'gpu' in navigator ? 'webgpu' : 'webgl';
    await tf.setBackend(backend);
    await tf.ready();
    console.log(`Using TensorFlow backend: ${backend}`);
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

    async function removeSilence(audioBuffer: AudioBuffer, sampleRate: number): Promise<AudioBuffer | null> {
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

    // Converts audio waveform into array of examples for VGGish
    async function waveformToExamples(data: AudioBuffer, sampleRate: number) {
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

// Runs feature extractor
async function runFeatureExtraction(audioBuffer: AudioBuffer) {

    const inputData = await preprocess(audioBuffer);

    try {
      const outputs = [];

      if (!inputData) {
          throw new Error('inputData is undefined');
      }

      const [batchSize, channels, height, width] = inputData.shape;
      console.log("starting feature extraction")

      for (let batch = 0; batch < batchSize; batch++) {
          // Assuming input_data is a 3D tensor (similar to permute(2, 1, 0) in Python)
          const inputDataBatch = inputData.slice([batch, 0, 0, 0], [1, 1, height, width]);

          const inputDataTfjs = inputDataBatch.transpose([0, 2, 1, 3]).reshape([1, 64, 96]);

          // Run inference using the TensorFlow.js model
          //const outputTensor = vggishModel.execute(inputDataTfjs);
          const outputTensor = await vggishModel.predict(inputDataTfjs);

          // Convert the output tensor to a flat array
          let outputArray;
          if (outputTensor instanceof tf.Tensor) {
              outputArray = Array.from(await outputTensor.data());
          } else if (Array.isArray(outputTensor) && outputTensor.length > 0) {
              outputArray = Array.from(await outputTensor[0].data());
          } else {
              throw new Error('Unexpected output from model execution');
          }

          // Push the output to the list
          outputs.push(outputArray);
      }
      console.log("outputs obtained")
      console.log(outputs)

      const outputsTensor = tf.tensor(outputs);
      const flattenedOutputsTensor = outputsTensor.reshape([-1, 128]);

      const pprocOutputTensor = postprocessorModel.predict(flattenedOutputsTensor);

      const postProcessedOutput = (pprocOutputTensor as tf.Tensor).arraySync();
      return postProcessedOutput;  
    } catch (error) {
      console.error('Error during inference:', error);
      return null;
    }
}

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

async function runVoiceModel(input: tf.Tensor): Promise<number> {
    const vnvOut = voiceModel.predict(input);

    const vnvPredictionsTensor = tf.argMax((vnvOut as tf.Tensor), 1);
    const vnvPredictionsArray = await vnvPredictionsTensor.array();

    const vnvDetectedClass = mode(vnvPredictionsArray as number[]);

    return vnvDetectedClass;
}

async function runReverbModel(input: tf.Tensor): Promise<number> {
    const reverbOut = reverbModel.predict(input);

    const reverbPredictionsTensor = tf.argMax((reverbOut as tf.Tensor), 1);
    const reverbPredictionsArray = await reverbPredictionsTensor.array();
    const reverbDetectedClass = mode(reverbPredictionsArray as number[]);

    return reverbDetectedClass;
}

// Runs the classifier model
async function runClassifier(audioBuffer: AudioBuffer) {
    const vggishOut = await runFeatureExtraction(audioBuffer)
    console.log(vggishOut)
    // const classifiersInputTensor = tf.tensor(vggishOut as number[][]).reshape([-1, 128]);

    // const vnvDetectedClass = await runVoiceModel(classifiersInputTensor);
    let voice_result: string = "singing voice";

    // if (vnvDetectedClass) {
    //   if (vnvDetectedClass === 2) {
    //       voice_result = "spoken voice";
    //   } else {
    //       voice_result = "no voice";
    //   }
    // }

    let messages: string[] = [`Detected ${voice_result}`];

    // if(vnvDetectedClass === 0) {
    //   const reverbDetectedClass = await runReverbModel(classifiersInputTensor)
      
    //   let reverb_result: string = reverbDetectedClass ? "reverb" : "no reverb";
    //   messages.push(`Detected ${reverb_result}`);
    // }

    return messages
}

export {init, runClassifier}
