clc
clear all

outFolder = fullfile('caltech101');
rootFolder = fullfile(outFolder, '101_ObjectCategories');

categories = {'airplanes', 'rhino', 'ferry', 'laptop', 'brain', 'wheelchair', 'umbrella', 'wild_cat', 'watch', 'dolphin'}; %Categories on which we are going to work with (can be expanded or changed

imds = imageDatastore(fullfile(rootFolder,categories), 'LabelSource','foldernames'); %Images of the categories we chose are now here with appropriate labels

tbl = countEachLabel(imds); %Count the number of images in each category
n = min(tbl{:,2}); %Find the minimum number of images in our categories

imds = splitEachLabel(imds, n, 'randomize'); %We choose n random images to have same number of images in each category

tbl = countEachLabel(imds);

airplanes = find(imds.Labels == 'airplanes', 1);
rhino = find(imds.Labels == 'rhino', 1);
ferry = find(imds.Labels == 'ferry', 1);
laptop = find(imds.Labels == 'laptop', 1);
brain = find(imds.Labels == 'brain', 1);
wheelchair = find(imds.Labels == 'wheelchair', 1);
umbrella = find(imds.Labels == 'umbrella', 1);
wild_cat = find(imds.Labels == 'wild_cat', 1);
watch = find(imds.Labels == 'watch', 1);
dolphin = find(imds.Labels == 'dolphin', 1);


net = resnet50(); %Net is a model which has been pretrained on the data set of images we provide

% figure(1)
% plot(net)
% title('Architecture of ResNet50')
% set(gca, 'YLim', [150 170])

net.Layers(1);
net.Layers(end);

numel(net.Layers(end).ClassNames);
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize'); %We split the images in categories to train and test the neural network

imageSize = net.Layers(1).InputSize; %gives us the required size of the input images

AugmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb'); %resize and covert any grayscale image into rbg image
AugmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

w1 = net.Layers(2).Weights; %w1 is a matrix. We need to convert it in image
w1 = mat2gray(w1); %Conversion

% figure(2)
% montage(w1)
% title('First Convolutional Layer Weight')

featureLayer = 'fc1000';
trainingFeatures = activations(net, AugmentedTrainingSet, ...
    featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLabels, 'Learner', 'Linear', ...
    'Coding', 'onevsall', 'ObservationsIn', 'columns'); %Returns a trained model and stores it

testFeatures = activations(net, AugmentedTestSet, ...
    featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

predictLabels = predict(classifier, testFeatures, ...
    'ObservationsIn', 'columns'); %Returns a vector of predicted class labels based on trained classifiers

testLabels = testSet.Labels;
confMat = confusionmat(testLabels, predictLabels);

confMat = bsxfun(@rdivide, confMat, sum(confMat,2)); %Convert values of confusion matrix in percentage (1 is 100%)

mean(diag(confMat)); %Accuracy
%% 
% Now we classify the images
newImage = imread(fullfile('test11.jpg')); %Load image in png or jpg form

ds = augmentedImageDatastore(imageSize, newImage, ...
    'ColorPreprocessing', 'gray2rgb');

imageFeatures = activations(net, ds, ...
    featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

label = predict(classifier, imageFeatures, ...
    'ObservationsIn', 'columns');

sprintf('The loaded image belongs to the %s class', label)
