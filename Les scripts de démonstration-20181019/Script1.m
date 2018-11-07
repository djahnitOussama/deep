% TD_Deep Learning   Master 2 EEA       2018
%                                       
% d�monstration utilisation d'un r�seau d�j� entrain�
% en temps r�el sur un flux vid�o
% 
% pour des explications plus compl�tes voir aussi les exemples de Matlab 
% en particulier, 'Classify Webcam Images Using Deep Learning'


%% Configuration

if ~exist('camera')
    camera = webcam;        % connexion de la cam�ra
end
 
net=alexnet;                % chargement du r�seau Deep
%   alexnet est une architecture tr�s populaire et libre disponible qui a
%   gagn� le concours ILSVRC en 2012 avec un taux d'erreur de 15,3% (m�thode top-5)

inputSize = net.Layers(1).InputSize(1:2);
%   taille de la couche d'entr�e du r�seau

%% Classification des frames vid�o

figure(1);clf
keepRolling = true;
set(gcf,'CloseRequestFcn','keepRolling = false; closereq');

while keepRolling                       % tant que l'utilisateur ne ferme pas la figure
    im = snapshot(camera);              % acquisition de l'image cam�ra

    im1 = imresize(im,inputSize);       % ajustement de la taille
    [label,score] = classify(net,im1);  % r�ponse du r�seau de neurones
    
    % affichage de l'image et de la r�ponse
    imshow(im);
    text(250,-10,char(label),'FontSize',14);    
    text(450,-10,num2str(max(score),'score : %4.2f'),'FontSize',12);    
    drawnow
end

clear camera