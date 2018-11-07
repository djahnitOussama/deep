% TD_Deep Learning   Master 2 EEA       2018
%                                       
% démonstration utilisation d'un réseau déjà entrainé
% détail du fonctionnement sur une image 
% 


%% Choix du réseau préentrainé
%%net=alexnet;
% alexnet est une architecture très populaire et libre disponible qui a
% gagné le concours ILSVRC en 2012 avec un taux d'erreur de 15,3 % (méthode top-5)


netNew.Layers      % affichage du détail de l'architecture
%   le réseau est composé de 5 couches convolutionnelles et 3 couches
%   entièrement connecté


inputSize = netNew.Layers(1).InputSize(1:2);
%   taille de la couche d'entrée du réseau

%% Test de réponse sur première image

repertoire='C:\Users\TEMP\Downloads\Caltech101Partiel\Caltech101Partiel';
im=imread(fullfile(repertoire, 'revolver','image_0002.jpg'));

figure(1),clf,      % affichage de l'image
imshow(im);

im1 = imresize(im,inputSize);       % ajustement de la taille
[label,score] = classify(netNew,im1);  % réponse du réseau de neurones

disp([label, num2str(max(score))])


figure(2),clf
plot(score,'*r')
% axis([80 120 -0.1 1.1])   % pour zoomer

% [valMax, indiceMax]=max(score)

%% Test de réponse d'une autre image

im=imread(fullfile(repertoire, 'revolver','image_0001.jpg'));

figure(1),clf,      % affichage de l'image
imshow(im);

im1 = imresize(im,inputSize);       % ajustement de la taille
[label,score] = classify(netNew,im1);  % réponse du réseau de neurones

disp([label, num2str(max(score))])

figure(2),clf
plot(score,'*r'),axis([0 1000 0 1])
% axis([80 120 -0.1 1.1])   % pour zoomer

% réponse top-5
[scoreT iT]=maxk(score,5);
labels5=netNew.Layers(end).ClassNames(iT)

%% réponse pour toutes les images du répertoire 'toucan'

% collecte de toutes les images
%images=zeros(227,227,3,100);
%%  imageName=['image' num2str(i,'%03i') '.jpg'];
   % im=imread(fullfile(repertoire, 'revolver',imageName));
    %%end

[labels,scores] = classify(netNew,images);  % réponse du réseau de neurones

% vérification des réponses
a=find(labels ~= 'revolver');
fprintf('nombre d''erreurs : %i sur %i\n',numel(a),numel(labels))

% affichage des erreurs
figure(3),clf
for i=1:4
    subplot(2,2,i)
    imshow(images(:,:,:,a(i))/255);
    text(20,-10,char(labels(a(i))),'FontSize',12);
    text(120,-10,num2str(max(score(a(i),:)),'score : %4.2f'),'FontSize',10);
end

figure(2),clf
plot(scores(a(1),:),'x')

%% réponse pour les 6 catégories d'oiseaux
% avec technique du datastore

inputSize = [227 227];   % net.Layers(1).InputSize
imds=imageDatastore(repertoire,'IncludeSubfolders',true, 'LabelSource','foldernames');
auimds = augmentedImageDatastore(inputSize,imds,'ColorPreprocessing','gray2rgb');
% détermination du nombre de catégories et affichage de quelques exemples
% (1 catégorie par ligne)
numClasses = numel(categories(imds.Labels))
countEachLabel(imds)

%net=alexnet;

% ajustement automatique de taille lors de la lecture des images dans le
% imagedatastore



% test  de classification
[labels,scores] = classify(netNew,auimds);

a=find(labels ~= imds.Labels);
fprintf('nombre d''erreurs : %i sur %i\n',numel(a),numel(labels))

b=contains([categories(imds.Labels)],netNew.Layers(end).ClassNames)