export interface ModelPaths {
    vggishModelPath: string;
    postprocessorModelPath: string;
    voiceModelPath: string;
    reverbModelPath: string;
}

export const vggishParams = {
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