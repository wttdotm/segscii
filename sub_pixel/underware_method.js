const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');

// ASCII character set (from dark to light)
const asciiChars = ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.'];

function imageToAscii(imagePath, outputWidth) {
    return loadImage(imagePath).then((image) => {
        const canvas = createCanvas(image.width, image.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const aspectRatio = image.height / image.width;
        const outputHeight = Math.round(outputWidth * aspectRatio);

        return _cpaToString(imageData.data, image.width, 
            { x: Math.ceil(image.width / outputWidth), y: Math.ceil(image.height / outputHeight) },
            { x: outputWidth, y: outputHeight },
            asciiChars.map((char, index) => ({
                letter: char,
                bottom: index * (255 / (asciiChars.length - 1)),
                top: (index + 1) * (255 / (asciiChars.length - 1))
            }))
        );
    });
}

function _cpaToString(cpa, width, charSize, blocks, colors) {
    var reducedPixelArray = [];
    var blockArray = [];
    var letters = [];

    // Convert RGB(A) to grayscale
    for (var b = 0, c = cpa.length; b < c; b += 4) {
        reducedPixelArray.push(Math.max(cpa[b], cpa[b + 1], cpa[b + 2]));
    }

    // Calculate average luminance for each character block
    for (var h = 0; h < blocks.y; h++) {
        for (var i = 0; i < blocks.x; i++) {
            var pixelsInBlock = [];
            var baseOffset = h * charSize.y * width + i * charSize.x;
            for (var j = 0; j < charSize.y; j++) {
                for (var k = 0; k < charSize.x; k++) {
                    var currentOffset = baseOffset + (j * width + k);
                    pixelsInBlock.push(reducedPixelArray[currentOffset]);
                }
            }
            blockArray.push(Math.round(pixelsInBlock.reduce((a, b) => a + b) / (charSize.x * charSize.y)));
        }
    }

    // Convert luminance values to ASCII characters
    for (var l = 0, m = blockArray.length; l < m; l++) {
        letters.push(colors.find(el => blockArray[0] <= el.top && blockArray[0] >= el.bottom).letter);
        blockArray.shift();
        if ((l + 1) % blocks.x === 0) {
            letters.push('\n');
        }
    }
    return letters.join('');
}

// Usage
const imagePath = './test_tree2.jpg';
const outputWidth = 100;  // Adjust this to change the width of your ASCII art

imageToAscii(imagePath, outputWidth)
    .then(asciiArt => console.log(asciiArt))
    .catch(err => console.error('Error:', err));