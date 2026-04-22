clear; close all; clc;
result_path = './results/';
OITNN_path = './other_models/Code4OITNN_v0.1/Functions4OITNN';
data_path = './results/tensorized_data';

addpath('./other_models/Code4OITNN_v0.1/Functions4OITNN');
addpath('./results/');
%%
model_keys = {'RTD_OITNN_O', 'RTD_OITNN_L', 'RTD_TNN'};
isMethodOn=[1 1 1];
smd_data = load([data_path '/tensorized_smd.mat']);
save([result_path 'smd_matlab_alg_results.mat'], 'model_keys');
%%
RTD_OITNN_O.name = 'RTD:OITNN-O';
for m_id=smd_data.machine_ids
    for ch_id = 1:smd_data.num_channels(m_id)
        m_str = ['m' int2str(m_id)];
        ch_str = ['ch' int2str(ch_id)];
        fprintf("Running RTD:OITNN-O for SMD machine %d channel %d \n", m_id, ch_id);
        Y = smd_data.data.(m_str).(ch_str);
        var_est = smd_data.var_est.(m_str).(ch_str);
        tic
        memo = run_OITNN_O(Y, var_est);
        toc
        RTD_OITNN_O.(m_str).(ch_str).Lhat = memo.Lhat;
        RTD_OITNN_O.(m_str).(ch_str).Shat = memo.Shat;
    end
end
save([result_path 'smd_matlab_alg_results.mat'], "RTD_OITNN_O", "-append");

RTD_OITNN_L.name = 'RTD:OITNN-L';
for m_id=smd_data.machine_ids
    for ch_id = 1:smd_data.num_channels(m_id)
        m_str = ['m' int2str(m_id)];
        ch_str = ['ch' int2str(ch_id)];
        fprintf("Running RTD:OITNN-L for SMD machine %d channel %d \n", m_id, ch_id);
        Y = smd_data.data.(m_str).(ch_str);
        var_est = smd_data.var_est.(m_str).(ch_str);
        tic
        memo = run_OITNN_O(Y, var_est);
        toc
        RTD_OITNN_L.(m_str).(ch_str).Lhat = memo.Lhat;
        RTD_OITNN_L.(m_str).(ch_str).Shat = memo.Shat;
    end
end
save([result_path 'smd_matlab_alg_results.mat'], "RTD_OITNN_L", "-append");
%%
RTD_TNN.name = 'RTD:TNN';
for m_id=smd_data.machine_ids
    for ch_id = 1:smd_data.num_channels(m_id)
        m_str = ['m' int2str(m_id)];
        ch_str = ['ch' int2str(ch_id)];
        fprintf("Running RTD:TNN for SMD machine %d channel %d \n", m_id, ch_id);
        Y = smd_data.data.(m_str).(ch_str);
        var_est = smd_data.var_est.(m_str).(ch_str);
        tic
        memo = run_RTDTNN(Y, var_est);
        toc
        RTD_TNN.(m_str).(ch_str).Lhat = memo.Lhat;
        RTD_TNN.(m_str).(ch_str).Shat = memo.Shat;
    end
end
save([result_path 'smd_matlab_alg_results.mat'], "RTD_TNN", "-append");

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
    opts.verbose=1;
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
    Yt = permute(Y, [1,2,4,3]);
    permuted_sz = size(Yt);
    T = prod(permuted_sz(3:4));
    Yt = reshape(Yt, [permuted_sz(1:2), T]);
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
    memo.Lhat = permute(reshape(memo.Lhat, permuted_sz), [1,2,4,3]);
    memo.Shat = permute(reshape(memo.Shat, permuted_sz), [1,2,4,3]);
end