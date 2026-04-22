clear; close all; clc;
result_path = './results/';
OITNN_path = './other_models/Code4OITNN_v0.1/Functions4OITNN';
TRPCA_path = './other_models/TRPCA';
data_path = './results/tensorized_data/hsi_abu/';

addpath('./other_models/Code4OITNN_v0.1/Functions4OITNN');
addpath('./other_models/TRPCA');
addpath('./results/');
%%
model_keys = {'RTD_OITNN_O', 'RTD_OITNN_L', 'RTD_TNN', 'TRPCA'};
cat_length.airport = 4;
cat_length.beach = 4;
cat_length.urban = 5;
cat_strs = {'airport', 'beach', 'urban'};
% Saving the models keys to the results for convenience.
file_name = 'hsi_matlab_alg_results_v2.mat';
% save([result_path file_name], 'model_keys');
var_est = 0.1;
noise_added = false;
%%
RTD_OITNN_O.name = 'RTD:OITNN-O';
RTD_OITNN_O.var_est = var_est;
RTD_OITNN_O.noise_added = noise_added;
for cidx=1:3
    cat = cat_strs{cidx};
    for img_id=1:cat_length.(cat)
        img_str = int2str(img_id);
        hsi_data = load([data_path 'abu-' cat '-' img_str '.mat']);
        fprintf("Running RTD:OITNN-O for %s image %d \n", cat, img_id);
        Y = hsi_data.data;
        if noise_added
            Yn = Y + sqrt(var_est)*randn(size(Y));
        else
            Yn = Y;
        end
        tic
        memo = run_OITNN_O(Yn, var_est);
        toc
        RTD_OITNN_O.(cat).(['im_' img_str]).Lhat = memo.Lhat;
        RTD_OITNN_O.(cat).(['im_' img_str]).Shat = memo.Shat;
    end
end
save([result_path file_name], "RTD_OITNN_O", '-append');
%%
RTD_OITNN_L.name = 'RTD:OITNN-L';
RTD_OITNN_L.var_est = var_est;
RTD_OITNN_L.noise_added = noise_added;
for cidx=1:3
    cat = cat_strs{cidx};
    for img_id=1:cat_length.(cat)
        img_str = int2str(img_id);
        hsi_data = load([data_path 'abu-' cat '-' img_str '.mat']);
        fprintf("Running RTD:OITNN-L for %s image %d \n", cat, img_id);
        Y = hsi_data.data;
        if noise_added
            Yn = Y + sqrt(var_est)*randn(size(Y));
        else
            Yn = Y;
        end
        tic
        memo = run_OITNN_L(Y, var_est);
        toc
        RTD_OITNN_L.(cat).(['im_' img_str]).Lhat = memo.Lhat;
        RTD_OITNN_L.(cat).(['im_' img_str]).Shat = memo.Shat;
    end
end
save([result_path file_name], "RTD_OITNN_L", "-append");
%%
RTD_TNN.name = 'RTD:TNN';
RTD_OITNN_L.var_est = var_est;
RTD_OITNN_L.noise_added = noise_added;
for cidx=1:3
    cat = cat_strs{cidx};
    for img_id=1:cat_length.(cat)
        img_str = int2str(img_id);
        hsi_data = load([data_path 'abu-' cat '-' img_str '.mat']);
        fprintf("Running RTD:TNN for %s image %d \n", cat, img_id);
        Y = hsi_data.data;
        if noise_added
            Yn = Y + sqrt(var_est)*randn(size(Y));
        else
            Yn = Y;
        end
        tic
        memo = run_RTDTNN(Yn, var_est);
        toc
        RTD_TNN.(cat).(['im_' img_str]).Lhat = memo.Lhat;
        RTD_TNN.(cat).(['im_' img_str]).Shat = memo.Shat;
    end
end
save([result_path file_name], "RTD_TNN", "-append");
%%
TRPCA.name = 'TRPCA';
for cidx=1:3
    cat = cat_strs{cidx};
    for img_id=1:cat_length.(cat)
        img_str = int2str(img_id);
        hsi_data = load([data_path 'abu-' cat '-' img_str '.mat']);
        fprintf("Running TRPCA for %s image %d \n", cat, img_id);
        Y = hsi_data.data;
        if noise_added
            Yn = Y + sqrt(var_est)*randn(size(Y));
        else
            Yn = Y;
        end
        tic
        memo = run_TRPCA(Y);
        toc
        TRPCA.(cat).(['im_' img_str]).Lhat = memo.Lhat;
        TRPCA.(cat).(['im_' img_str]).Shat = memo.Shat;
    end
end
save([result_path file_name], "TRPCA", "-append");

%% OITNN_O_settings
function memo = run_OITNN_O(Y, var_est)
    sigma = sqrt(var_est);

    sz = size(Y);
    K = length(sz);
    D = prod(sz);
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
    opts.para.lambdaL=ldaO;
    opts.para.lambdaS=muO;
    opts.para.alpha=alphaL;
    opts.para.rho=rho;
    opts.para.nu=nu;
    opts.para.vW=w;
    opts.MAX_ITER_OUT=500;
    opts.MAX_RHO=1e10;
    opts.MAX_EPS=1e-6;
    opts.verbose=0;
    %+++++Algorithm options+++++
    %+++++construct memo+++++
    memo=h_construct_memo_v2(opts);
    memo.truthL=Y;
    memo.truthS=Y;
    opts.showImg=0;
    %+++++construct memo+++++
    %++++++++++++++Run++++++++++++++
    t=clock;
    memo=f_rtd_OITNN_O(obs,opts,memo);
    t=etime(clock,t);
end

%% RTD:OITNN_L_settings
function memo = run_OITNN_L(Y, var_est)
    sz = size(Y);
    K = length(sz);
    D = prod(sz);
    sigma = sqrt(var_est);
    alphaL=max( abs(Y(:)));
    alphaS=alphaL;
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
    opts.obs=obs;
    opts.para.lambdaL=lamL;
    opts.para.lambdaS=lamS;
    opts.para.alpha=alphaL;
    opts.para.rho=rho;
    opts.para.nu=nu;
    opts.para.vW=vV;
    opts.MAX_ITER_OUT=300;
    opts.MAX_RHO=1e10;
    opts.MAX_EPS=1e-6;
    opts.verbose=0;
    opts.showImg=0;
    %+++++Algorithm options+++++
    
    %+++++construct memo+++++
    memo = h_construct_memo_v2(opts);
    memo.truthL=Y;
    memo.truthS=Y;
    %+++++construct memo+++++
    
    %++++++++++++++Run++++++++++++++
    t=clock;
    memo=f_rtd_OITNN_L(obs,opts,memo);
    t=etime(clock,t);
    %++++++++++++++Run++++++++++++++
end
%% RTD:TNN Settings

function memo = run_RTDTNN(Y, var_est)
    og_sz = size(Y);
    Yt = Y;
    sz = size(Yt);
    K = length(sz);
    D = prod(sz);
    alphaL=max( abs(Y(:)));
    alphaS=alphaL;
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
    optsTNN.verbose=0;
    optsTNN.showImg=0;
    %+++++Algorithm options+++++
    
    %+++++construct memo+++++
    memo=h_construct_memo_v2(optsTNN);
    memo.truthL=Yt;
    memo.truthS=Yt;
    %+++++construct memo+++++
    
    %++++++++++++++Run++++++++++++++
    t=clock;
    memo=f_rtd_TNN(obs,optsTNN,memo);
    t=etime(clock,t);
    %++++++++++++++Run++++++++++++++
end

%% TRPCA Settings:
function memo = run_TRPCA(Y)
    sz = size(Y);
    K = length(sz);
    D = prod(sz);
    n1 = sz(1);
    n2 = sz(2);
    n3 = sz(3);

    %++++++Model Parameters++++++
    lambda = 1/sqrt(n3*max(n1,n2));
    %++++++Model Parameters++++++
    %++++++Algorithm Paramters+++++ 
    opts.tol = 1e-8;
    opts.rho = 1.1;
    opts.mu = 1e-4;
    opts.DEBUG = 0;
    %++++++Algorithm Paramters+++++ 
    
    % +++Observation+++
    obs.tY=Y;
    % +++Observation+++
    
    %++++++++++++++Run++++++++++++++
    t=clock;
    [Lhat,Shat] = trpca_tnn(Y,lambda,opts);
    t=etime(clock,t);
    %++++++++++++++Run++++++++++++++
    memo.Lhat = Lhat;
    memo.Shat = Shat;
end
