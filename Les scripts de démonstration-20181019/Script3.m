% TD_Deep Learning   Master 2 EEA       2018
%                                       
% création d'un nouveau réseau avec transfert d'apprentissage
% et apprentissage pour ajusement sur les nouvelles images
% 
% 
% pour plus de détails, voir l'exemple de démonstration Matlab
% "Create Simple Deep Learning Network for Classification"

%% Préparation des images
% chargement des images 
repertoire='C:\Users\TEMP\Downloads\Caltech101Partiel\Caltech101Partiel';
imds=imageDatastore(repertoire,'IncludeSubfolders',true, 'LabelSource','foldernames');

numClasses = numel(categories(imds.Labels))

%% Création d'un réseau de neurones
layers = [
    imageInputLayer([227 227 3])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(numClasses)      % 6 classes à reconnaitre
    softmaxLayer
    classificationLayer];

% ajustement automatique de taille lors de la lecture des images dans le
% imagedatastore
inputSize =  layers(1).InputSize(1:2);




%% Adaptation du réseau 

options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');


% répartition des exemples disponibles en 2 lots : 
% ensemble d'apprentissage et ensemble de test  (indépendants)
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');

auimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,'ColorPreprocessing','gray2rgb');
auimdsTest = augmentedImageDatastore(inputSize,imdsTest,'ColorPreprocessing','gray2rgb');

% Apprentissage
netNew = trainNetwork(auimdsTrain,layers,options);

% test sur images restantes
[labels,scores] = classify(netNew,auimdsTest);
taux=sum(labels==imdsTest.Labels)/numel(labels)*100;
fprintf('taux de classification correcte après entrainement = %5.1f%%\n',taux);

