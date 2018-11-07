% TD_Deep Learning   Master 2 EEA       2018
%                                       
% cr�ation d'un nouveau r�seau avec transfert d'apprentissage
% et apprentissage pour ajusement sur les nouvelles images
% 
% pour plus de d�tails, voir l'exemple de d�monstration Matlab
% "Transfer Learning Using AlexNet"
%% Pr�paration des images
% chargement des images 
repertoire='C:\Users\TEMP\Downloads\ImagesTest';
imds=imageDatastore(repertoire,'IncludeSubfolders',true, 'LabelSource','foldernames');
numClasses = numel(categories(imds.Labels));


% r�partition des exemples disponibles en 2 lots : 
% ensemble d'apprentissage et ensemble de test  (ind�pendants)

[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');


%% le r�seau Deep pr�entrain� sur ImageNet (AlexNet)

net=netNew;

% ajustement automatique de taille lors de la lecture des images dans le
% imagedatastore
inputSize = [227 227];   % net.Layers(1).InputSize
auimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,'ColorPreprocessing','gray2rgb');
auimdsTest = augmentedImageDatastore(inputSize,imdsTest,'ColorPreprocessing','gray2rgb');



%% Adaptation du r�seau : 
% Transfert des poids, cr�ation d'une nouvelle couche de classification et entrainement

layers = [
    net.Layers(1:end-3);   % transfert des premi�res couches
                            % couches de classification
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',4e-5, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');


% Apprentissage
netTransfer = trainNetwork(auimdsTrain,layers,options);

% test sur images restantes
[YPred,scores] = classify(netTransfer,auimdsTest);
taux=sum(YPred==imdsTest.Labels)/numel(YPred)*100;
fprintf('taux de classification correcte apr�s entrainement = %5.1f%%\n',taux);

% v�rification top-2
a=find(YPred~=imdsTest.Labels);
[m, im]=maxk(scores,2,2);
