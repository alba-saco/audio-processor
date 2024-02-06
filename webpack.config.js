const path = require('path');
const pathBrowserify = require('path-browserify');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const TerserPlugin = require("terser-webpack-plugin");

module.exports = {
    mode: 'development',
    entry: './src/audioProcessor.js',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'bundle.js',
    },
    plugins: [
        new CopyWebpackPlugin({
            patterns: [
                { from: 'node_modules/onnxruntime-web/dist/**/*.wasm', to: 'onnxruntime-web/dist/[name][ext]' },
                { from: 'node_modules/@tensorflow/tfjs-backend-wasm/dist/*.wasm', to: '@tensorflow/tfjs-backend-wasm/[name][ext]'}
            ],
        }),
    ],
    module: {
      rules: [
        {
          test: /\.js$/,
          use: 'babel-loader',
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
      ],
    },
    resolve: {
        extensions: ['.js', '.wasm'],
        fallback: {
            "path": require.resolve("path-browserify"),
        },
        modules: [path.resolve(__dirname, 'node_modules')],
    },
    optimization: {
      minimizer: [
        new TerserPlugin({
          terserOptions: {
            compress: {
              unused: false,
            }
          }
        })
      ]
    }
  };