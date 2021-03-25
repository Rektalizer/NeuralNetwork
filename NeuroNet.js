/*
My first neural network is designed for an identification task from learning video I watched. We have a set of
flowers growing at the flowerbed. Two types of flowers are there - flowers with blue pelts and flowers with red pelts.
We measure length and width of one pelt on each individual flower and use it as our initial data. But after all
measuring we found out that we wrote down one flower without assigning it color. For us it's to time-consuming to
remeasure all flowers so we build a simple neural network which will automatically assign the unknown flower to its
true color.
*/

// Training data. Goes by [Length of pelt, width of pelt, color of pelt (Where 0 = blue and 1 = red)]
// Data for red flowers
let blueData1 = [0.5, 2, 0];
let blueData2 = [2, 1.5, 0];
let blueData3 = [2, 0.5, 0];
let blueData4 = [3, 1, 0];

// Data for blue flowers
let redData1 = [3, 1.5, 1];
let redData2 = [3.5, 0.5, 1];
let redData3 = [4, 1.5, 1];
let redData4 = [5.5, 1, 1];

// Data for flower which color we don't know (data we want to find)
let goalData = [4.5,  1, "Should be 1 here"];

let all_points = [blueData1, blueData2, blueData3, blueData4, redData1, redData2, redData3, redData4];

// Defining sigmoid activation function
function sigmoid(x) {
    return 1/(1+Math.exp(-x));
}

// Training
function train() {
    /* Weight initialization (Here we are starting training with assigning random values to weights and bias for our
    neural network to compute from) */
    let w1 = Math.random()*.2-.1;
    let w2 = Math.random()*.2-.1;
    let b = Math.random()*.2-.1;
    // Learning rate is basically a fraction of update we use to not overextend weights and bias in any direction
    let learning_rate = 0.2;
    // Executing learning iterations
    for (let iter = 0; iter < 50000; iter++) {
        // Picking random flower(point) from our data
        let random_idx = Math.floor(Math.random() * all_points.length);
        let point = all_points[random_idx];
        let target = point[2]; // our target part of data (color) picked from 3d coordinate of a point

        // Feedforward process
        let z = w1 * point[0] + w2 * point[1] + b; // Our main function [Prediction = weight_1 * data_1 + weight_2 * data_2]
                                                  // Where data_1 is pelt length and data_2 is pelt width
        let pred = sigmoid(z);                   // Then we 'activate' the function with sigmoid

        // Now we compare the model prediction with the target with "square error cost function"
        let cost = (pred - target) ** 2;

        // Now we find the slope of the cost with respect to each parameter (w1, w2, b)
        // Bring derivative of cost through square error function (Power rule) with respect to prediction
        let dcost_dpred = 2 * (pred - target);

        // Bring derivative of prediction through sigmoid with respect to Z
        // Derivative of sigmoid we take for derivative formula for sigmoid: [d/dz sigmoid(z) = sigmoid(z)*(1-sigmoid(z))]
        let dpred_dz = sigmoid(z) * (1-sigmoid(z));

        // Bring derivative of Z with respect to weights and bias
        let dz_dw1 = point[0];
        let dz_dw2 = point[1];
        let dz_db = 1;

        // Now we can get the partial derivatives (our update) using the chain rule
        /* We're bringing how the cost changes through each function, first through the square, then through the sigmoid
        and finally whatever is multiplying our parameter of interest becomes the last part */
        let dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1;
        let dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2;
        let dcost_db =  dcost_dpred * dpred_dz * dz_db;

        // Here we update our parameters by decrementing them with learning rate multiplied by update
        w1 -= learning_rate * dcost_dw1;
        w2 -= learning_rate * dcost_dw2;
        b -= learning_rate * dcost_db;
    }

    return {w1: w1, w2: w2, b: b};
}
