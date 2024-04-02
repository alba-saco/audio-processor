const fs = require('fs');
const path = require('path');

// Define the directory containing your model shards
const modelDirectory = './models/vggish-tfjs/';

// Function to get all files in a directory
function getAllFiles(dirPath, arrayOfFiles) {
    const files = fs.readdirSync(dirPath);

    arrayOfFiles = arrayOfFiles || [];

    files.forEach(function (file) {
        if (fs.statSync(dirPath + '/' + file).isDirectory()) {
            arrayOfFiles = getAllFiles(dirPath + '/' + file, arrayOfFiles);
        } else {
            arrayOfFiles.push(path.join(dirPath, '/', file));
        }
    });

    return arrayOfFiles;
}

// Get all files in the model directory
const allFiles = getAllFiles(modelDirectory);

// Filter out JSON files (model.json) and sort other files
const modelChunks = allFiles.filter(file => !file.endsWith('model.json')).sort();

// Create an object to store model chunk paths
const modelJsonData = {
    modelPath: 'model.json',
    chunks: modelChunks
};

// Write the JSON object to a file
fs.writeFileSync(path.join(modelDirectory, 'model_chunks.json'), JSON.stringify(modelJsonData, null, 4));

console.log('Model chunks JSON file generated successfully!');