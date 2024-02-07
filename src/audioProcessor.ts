// ... (other code remains unchanged)

async function preprocess(audioBuffer: AudioBuffer): Promise<tf.Tensor | null> {
    // ... (existing code)

    try {
        const spectrogram = await waveformToExamples(audioBuffer, audioBuffer.sampleRate);

        if (spectrogram) {
            return spectrogram;
        } else {
            console.log("Error computing Log Mel Spectrogram.");
            return tf.zeros([1, 64, 96]); // Return a default tensor when computation fails
        }
    } catch (error) {
        console.error("Error in preprocess:", error);
        return tf.zeros([1, 64, 96]); // Return a default tensor when computation fails
    }
}

// ... (other code remains unchanged)
