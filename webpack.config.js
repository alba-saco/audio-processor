const path = require('path');
const pathBrowserify = require('path-browserify');
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = {
    mode: 'production',
    entry: './src/audioProcessor.ts',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'audioProcessor.js',
        library: 'AudioProcessor',
        libraryTarget: 'umd',
    },
    plugins: [
        new CopyWebpackPlugin({
            patterns: [
                // { from: 'node_modules/onnxruntime-web/dist/**/*.wasm', to: 'onnxruntime-web/dist/[name][ext]' },
                { from: 'node_modules/@tensorflow/tfjs-backend-wasm/dist/*.wasm', to: '@tensorflow/tfjs-backend-wasm/[name][ext]'},
                // { from: 'node_modules/onnxruntime-web/dist/ort.webgpu.min.js', to: 'onnxruntime-web/dist/' }
            ],
        }),
    ],
    module: {
      rules: [
        {
          test: /\.ts$/,
          use: 'ts-loader',
          exclude: /node_modules/,
        },
        {
          test: /\.d\.ts$/, // Handle declaration files
          use: 'raw-loader', // Use raw-loader to handle declaration files
          exclude: /node_modules/,
        },
        {
          test: /\.wasm$/,
          type: 'webassembly/async',
        },
        {
          test: /\.wasm$/i,
          type: 'javascript/auto',
          use: [
            {
              loader: 'file-loader',
            },
          ],
        },
        // { test:/models\/.*\.bin/, type:"asset/inline"},
        // { test:/models\/.*\.json/, type:"asset/source"}   
      ],
    },
    resolve: {
        extensions: ['.js', '.ts', '.wasm'],
        fallback: {
            "path": require.resolve("path-browserify"),
        },
        modules: [path.resolve(__dirname, 'node_modules')],
    },
    optimization: {
      chunkIds: 'total-size'
    }
  };