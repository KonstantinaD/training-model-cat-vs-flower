let flowerImages = [];
let catImages = [];

function preload() {
    for (let i = 0; i < 5000; i++) {
        flowerImages[i] = loadImage(`data/flower (${i + 1}).png`);
        catImages[i] = loadImage(`data/cat (${i + 1}).png`);
    }
}

let imageClassifier;

function setup() {
    // for (let i = 0; i < 10000; i++) {
    //     flowerImages[i] = loadImage(`data/flower (${i + 1}).png`);
    //     catImages[i] = loadImage(`data/cat (${i + 1}).png`);
    // }

    createCanvas(400, 400);
    // image(flowerImages[0], 0, 0, width, height);

const IMAGE_WIDTH = 64;
const IMAGE_HEIGHT = 64;
const IMAGE_CHANNELS = 4;

let options = {
    inputs: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    // outputs: ['label'],
    task: 'imageClassification',
    debug: true,
};

imageClassifier = ml5.neuralNetwork(options);

for (let i = 0; i < flowerImages.length; i++) {
    imageClassifier.addData({ image: flowerImages[i] }, { label: 'Flower' });
    imageClassifier.addData({ image: catImages[i] }, { label: 'Cat' });
}

imageClassifier.normalizeData();
imageClassifier.train({ epochs: 50 }, finishedTraining);
}

function finishedTraining() {
    console.log('finished training!');
    imageClassifier.save();
}