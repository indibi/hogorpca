clear; close all; clc;
result_path = './results/';
OITNN_path = './other_models/Code4OITNN_v0.1/Functions4OITNN';
data_path = './results/tensorized_data';

addpath('./other_models/Code4OITNN_v0.1/Functions4OITNN');
addpath('./results/');
%% 

methodList={'RTD:OITNN-O','RTD:OITNN-L','RTD:TNN'};
isMethodOn=[1 1 1];
smd_data = load('tensorized_smd.mat');
nyc_data = load('arrivals.mat');
nyc_var_est = 94.5344;

%%
% Y = smd_data.data.m1.ch1;
% var_est = smd_data.var_est.m1.ch1;
Y = nyc_data.Y;
sigma = sqrt(nyc_var_est);
%sigma = sqrt(var_est);
sz = size(Y); K = length(sz); D = prod(sz);
G = randn(sz)*sigma;
noise_dual_tnns = zeros(1,K);
noise_dual_linf = max(abs(G(:)));
for k=1:K
    noise_dual_tnns(k) = f_tensor_spectral_norm(f_3DReshape(G,k))*K;
end
alphaL=max( abs(Y(:)));
alphaS=alphaL;

%%  RTD: OITNN-O
iModel =1;

%++++++Model Parameters++++++
% The parameters may be not optimal
% Following Thm 3:
w = ones(1,K)/K;
alphaL=max( abs(Y(:)));
alphaS=alphaL;
ldaO = 2*sigma*(sz/w)/K^2;
muO = 8*sigma*sqrt(D) + 16*alphaS;
    
%++++++Algorithm Paramters+++++ 
rho=1e0; nu=1;
%++++++Algorithm Paramters+++++

% +++Observation+++
obs.tY=Y;
% +++Observation+++

%+++++Algorithm options+++++
optsO.para.lambdaL=ldaO;
optsO.para.lambdaS=muO;
optsO.para.alpha=alphaL;
optsO.para.rho=rho;
optsO.para.nu=nu;
optsO.para.vW=w;
optsO.MAX_ITER_OUT=500;
optsO.MAX_RHO=1e10;
optsO.MAX_EPS=1e-6;
optsO.verbose=1;
%+++++Algorithm options+++++
%+++++construct memo+++++
memoO=h_construct_memo_v2(optsO);
memoO.truthL=Y;
memoO.truthS=Y;
optsO.showImg=0;
%+++++construct memo+++++
%++++++++++++++Run++++++++++++++
t=clock;
memoO=f_rtd_OITNN_O(obs,optsO,memoO);
t=etime(clock,t);
%++++++++++++++Run++++++++++++++
Lhat = memoO.Lhat;
Shat = memoO.Shat;
save RTD-OITNN-O_nyc.mat Lhat Shat;
%%  RTD: OITNN-L
iModel =2;
if isMethodOn( iModel)

%++++++Model Parameters++++++
% The parameters may be not optimal
% Following Thm 4:
c=1.0;
vV = ones(1, K);
vV=vV/sum(vV);
lamL = c*sigma*max(sz/vV);
lamS=(8*sigma*sqrt(log(D)) +  16*K*alphaS);


%++++++Model Parameters++++++


%++++++Algorithm Paramters+++++   
rho=1e0; nu=1;
%++++++Algorithm Paramters+++++   

% +++Observation+++
obs.tY=Y;
% +++Observation+++

%+++++Algorithm options+++++
optsL.obs=obs;
optsL.para.lambdaL=lamL;
optsL.para.lambdaS=lamS;
optsL.para.alpha=alphaL;
optsL.para.rho=rho;
optsL.para.nu=nu;
optsL.para.vW=vV;
optsL.MAX_ITER_OUT=300;
optsL.MAX_RHO=1e10;
optsL.MAX_EPS=1e-6;
optsL.verbose=1;
optsL.showImg=0;
%+++++Algorithm options+++++

%+++++construct memo+++++
memoL=h_construct_memo_v2(optsL);
memoL.truthL=Y;
memoL.truthS=Y;
%+++++construct memo+++++

%++++++++++++++Run++++++++++++++
t=clock;
memoL=f_rtd_OITNN_L(obs,optsL,memoL);
t=etime(clock,t);
%++++++++++++++Run++++++++++++++
Lhat = memoL.Lhat;
Shat = memoL.Shat;
save RTD-OITNN-L_nyc.mat Lhat Shat;
end

%% RTD: TNN
Yt = reshape(Y, [7*24 53 81]);
Yt = permute(Yt, [3 2 1]);
iModel =3;
if isMethodOn( iModel)
    %++++++Model Parameters++++++
    CTNN=5e0;
    lamL=CTNN;
    lamS=lamL/sqrt(sz(1)*sz(3));
    %++++++Model Parameters++++++
    
    
    %++++++Algorithm Paramters+++++ 
    rho=1e-3; nu=1.1;
    %++++++Algorithm Paramters+++++ 
    
    % +++Observation+++
    obs.tY=Yt;
    % +++Observation+++
    
    %+++++Algorithm options+++++
    opts.obs=obs;
    optsTNN.para.lambdaL=lamL;
    optsTNN.para.lambdaS=lamS;
    optsTNN.para.alpha=alphaL;
    optsTNN.para.rho=rho;
    optsTNN.para.nu=nu;
    optsTNN.MAX_ITER_OUT=300;
    optsTNN.MAX_RHO=1e10;
    optsTNN.MAX_EPS=1e-4;
    optsTNN.verbose=1;
    optsTNN.showImg=0;
    %+++++Algorithm options+++++
    
    %+++++construct memo+++++
    memoTNN=h_construct_memo_v2(optsTNN);
    memoTNN.truthL=Yt;
    memoTNN.truthS=Yt;
    %+++++construct memo+++++
    
    %++++++++++++++Run++++++++++++++
    t=clock;
    memoTNN=f_rtd_TNN(obs,optsTNN,memoTNN);
    t=etime(clock,t);
    %++++++++++++++Run++++++++++++++

    Ltrpca = reshape(permute(memoTNN.Lhat, [3 2 1]), [24 7 53 81]);
    Strpca = reshape(permute(memoTNN.Shat, [3 2 1]), [24 7 53 81]);
end
