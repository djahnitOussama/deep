% TD_Deep Learning   Master 2 EEA       2018
%                                       
% démonstration utilisation d'un réseau déjà entrainé
% en temps réel sur un flux vidéo
% 
% pour des explications plus complètes voir aussi les exemples de Matlab 
% en particulier, 'Classify Webcam Images Using Deep Learning'


%% Configuration

if ~exist('camera')
    camera = webcam;        % connexion de la caméra
end
 
net=alexnet;                % chargement du réseau Deep
%   alexnet est une architecture très populaire et libre disponible qui a
%   gagné le concours ILSVRC en 2012 avec un taux d'erreur de 15,3% (méthode top-5)

inputSize = net.Layers(1).InputSize(1:2);
%   taille de la couche d'entrée du réseau

%% Classification des frames vidéo

figure(1);clf
keepRolling = true;
set(gcf,'CloseRequestFcn','keepRolling = false; closereq');

while keepRolling                       % tant que l'utilisateur ne ferme pas la figure
    im = snapshot(camera);              % acquisition de l'image caméra

    im1 = imresize(im,inputSize);       % ajustement de la taille
    [label,score] = classify(net,im1);  % réponse du réseau de neurones
    
    % affichage de l'image et de la réponse
    imshow(im);
    text(250,-10,char(label),'FontSize',14);    
    text(450,-10,num2str(max(score),'score : %4.2f'),'FontSize',12);    
    drawnow
end

clear camera