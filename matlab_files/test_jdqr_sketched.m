% Test jdqr_sketched_v1 vs jdqr_sym_mod
% NOTE: In sketched JD, M = (Theta*V)'*(Theta*A*V) is NOT symmetric even for
% symmetric A. This means the Ritz vectors from eig(M) are not orthonormal,
% which affects restart/deflation strategies.
%
% Uses diagonal preconditioning: M = diag(A)
%
clear all; clc;
noplots = 3;
fprintf('============================================================\n');
fprintf('Comparison: jdqr_sketched_v1 vs jdqr_sym_mod (SMALLEST eigenvalues)\n');
fprintf('With diagonal preconditioning\n');
fprintf('============================================================\n\n');

%% Common options
k = 3;
opts.tol = 1e-8;
opts.maxit = 300;
opts.disp = 0;  % Set to 1 for verbose output
opts.jmin = 30;
opts.jmax = 60;
opts.sigma = 'SR';  % smallest real eigenvalues

results = {};

%% =========================================================================
%% TEST 1: Si2
%% =========================================================================
fprintf('=== TEST 1: Si2 (n = 769) ===\n');
load('Si2.mat')
A = Problem.A;
A = full(A);

% Diagonal preconditioner
d = diag(A);
d(abs(d) < 1e-10) = 1;  % Avoid division by zero
opts.M = spdiags(1./d, 0, size(A,1), size(A,1));

true_eigs = sort(eig(A), 'ascend');
fprintf('True smallest 3: %.4f, %.4f, %.4f\n', true_eigs(1:3));
fprintf('Gap ratio: %.2f\n\n', (true_eigs(1)-true_eigs(2))/(true_eigs(2)-true_eigs(3)));

rng(42);
tic; [~, L1_ref, h1_ref] = jdqr_sym_mod(A, k, opts); t1_ref = toc;
L1_ref = diag(L1_ref);

% Sketched version
opts.sketch_size = opts.jmax*2;
rng(42);
tic; [~, L1_sk, h1_sk] = jdqr_sketched_v1(A, k, opts); t1_sk = toc;

fprintf('jdqr_sym:     %d eigs, %4d matvecs, %.3fs\n', length(L1_ref), h1_ref(end,3), t1_ref);
fprintf('  Eigenvalues: '); fprintf('%.8f  ', L1_ref); fprintf('\n');
fprintf('sketched:     %d eigs, %4d matvecs, %.3fs\n', length(diag(L1_sk)), h1_sk(end,3), t1_sk);
fprintf('  Eigenvalues: '); fprintf('%.8f  ', diag(L1_sk)); fprintf('\n');
results{1} = struct('name', 'Si2 (n = 769)', 'ref', h1_ref, 'sk', h1_sk, ...
    'true_eigs', true_eigs(1:k), 'eigs_ref', L1_ref, 'eigs_sk', diag(L1_sk));

%% =========================================================================
%% TEST 2: SiH4
%% =========================================================================
fprintf('\n=== TEST 2: SiH4 (n = 5041) ===\n');
load('SiH4.mat')
A = Problem.A;
A = full(A);

% Diagonal preconditioner
d = diag(A);
d(abs(d) < 1e-10) = 1;
opts.M = spdiags(1./d, 0, size(A,1), size(A,1));

true_eigs = sort(eig(A), 'ascend');
fprintf('True smallest 3: %.6f, %.6f, %.6f\n', true_eigs(1:3));
fprintf('Gap ratio: %.2f\n\n', (true_eigs(1)-true_eigs(2))/(true_eigs(2)-true_eigs(3)));

rng(42);
tic; [~, L2_ref, h2_ref] = jdqr_sym_mod(A, k, opts); t2_ref = toc;
L2_ref = diag(L2_ref);
rng(42);
tic; [~, L2_sk, h2_sk] = jdqr_sketched_v1(A, k, opts); t2_sk = toc;

fprintf('jdqr_sym:     %d eigs, %4d matvecs, %.3fs\n', length(L2_ref), h2_ref(end,3), t2_ref);
fprintf('  Eigenvalues: '); fprintf('%.8f  ', L2_ref); fprintf('\n');
fprintf('sketched:     %d eigs, %4d matvecs, %.3fs\n', length(diag(L2_sk)), h2_sk(end,3), t2_sk);
fprintf('  Eigenvalues: '); fprintf('%.8f  ', diag(L2_sk)); fprintf('\n');
results{2} = struct('name', 'SiH4 (n = 5041)', 'ref', h2_ref, 'sk', h2_sk, ...
    'true_eigs', true_eigs(1:k), 'eigs_ref', L2_ref, 'eigs_sk', diag(L2_sk));

%% =========================================================================
%% TEST 3: Na5
%% =========================================================================
fprintf('\n=== TEST 3: Na5 (n = 5832) ===\n');
load('Na5.mat')
A = Problem.A;
A = full(A);

% Diagonal preconditioner
d = diag(A);
d(abs(d) < 1e-10) = 1;
opts.M = spdiags(1./d, 0, size(A,1), size(A,1));

true_eigs = sort(eig(A), 'ascend');
fprintf('True smallest 3: %.6f, %.6f, %.6f\n', true_eigs(1:3));
fprintf('Gap ratio: %.4f\n\n', (true_eigs(1)-true_eigs(2))/(true_eigs(2)-true_eigs(3)));

rng(42);
tic; [~, L3_ref, h3_ref] = jdqr_sym_mod(A, k, opts); t3_ref = toc;
L3_ref = diag(L3_ref);
rng(42);
tic; [~, L3_sk, h3_sk] = jdqr_sketched_v1(A, k, opts); t3_sk = toc;

fprintf('jdqr_sym:     %d eigs, %4d matvecs, %.3fs\n', length(L3_ref), h3_ref(end,3), t3_ref);
fprintf('  Eigenvalues: '); fprintf('%.8f  ', L3_ref); fprintf('\n');
fprintf('sketched:     %d eigs, %4d matvecs, %.3fs\n', length(diag(L3_sk)), h3_sk(end,3), t3_sk);
fprintf('  Eigenvalues: '); fprintf('%.8f  ', diag(L3_sk)); fprintf('\n');
results{3} = struct('name', 'Na5 (n = 5832)', 'ref', h3_ref, 'sk', h3_sk, ...
    'true_eigs', true_eigs(1:k), 'eigs_ref', L3_ref, 'eigs_sk', diag(L3_sk));


%% =========================================================================
%% CONVERGENCE PLOTS
%% =========================================================================
figure('Position', [50, 50, 1400, 900]);

for i = 1:noplots
    % By iteration
    subplot(2, noplots, i);
    if ~isempty(results{i}.ref)
        semilogy(results{i}.ref(:,1), 'b-', 'LineWidth', 1.5); hold on;
    end
    if ~isempty(results{i}.sk)
        semilogy(results{i}.sk(:,1), 'r--', 'LineWidth', 1.5);
    end
    yline(opts.tol, 'k:', 'LineWidth', 1);
    title(results{i}.name);
    xlabel('Iteration');
    if i == 1, ylabel('Residual'); end
    if i == 1, legend('jdqr\_sym', 'sketched', 'Location', 'best'); end
    grid on;
    
    % By matvecs
    subplot(2, noplots, noplots+i);
    if ~isempty(results{i}.ref)
        semilogy(results{i}.ref(:,3), results{i}.ref(:,1), 'b-', 'LineWidth', 1.5); hold on;
    end
    if ~isempty(results{i}.sk)
        semilogy(results{i}.sk(:,3), results{i}.sk(:,1), 'r--', 'LineWidth', 1.5);
    end
    yline(opts.tol, 'k:', 'LineWidth', 1);
    xlabel('Matvecs');
    if i == 1, ylabel('Residual'); end
    grid on;
end

sgtitle('Top: by iteration, Bottom: by matvecs (blue=jdqr\_sym, red=sketched) - WITH DIAGONAL PRECOND');

%% =========================================================================
%% SUMMARY TABLE
%% =========================================================================
fprintf('\n============================================================\n');
fprintf('SUMMARY: Matvec comparison (with diagonal preconditioning)\n');
fprintf('============================================================\n');
fprintf('%-25s  %10s  %10s  %10s\n', 'Test', 'jdqr_sym', 'sketched', 'ratio');
fprintf('------------------------------------------------------------\n');
for i = 1:noplots
    mv_ref = results{i}.ref(end, 3);
    mv_sk = results{i}.sk(end, 3);
    fprintf('%-25s  %10d  %10d  %10.2f\n', results{i}.name, mv_ref, mv_sk, mv_sk/mv_ref);
end
fprintf('------------------------------------------------------------\n');

fprintf('\n============================================================\n');
fprintf('EIGENVALUE ACCURACY\n');
fprintf('============================================================\n');
for i = 1:noplots
    fprintf('\n%s:\n', results{i}.name);
    fprintf('  %-12s  %16s  %16s  %16s\n', 'Index', 'True', 'jdqr_sym', 'sketched');
    fprintf('  ----------------------------------------------------------------\n');
    n_true = length(results{i}.true_eigs);
    n_ref = length(results{i}.eigs_ref);
    n_sk = length(results{i}.eigs_sk);
    for j = 1:max([n_true, n_ref, n_sk])
        if j <= n_true
            true_str = sprintf('%.8f', results{i}.true_eigs(j));
        else
            true_str = '-';
        end
        if j <= n_ref
            ref_str = sprintf('%.8f', results{i}.eigs_ref(j));
        else
            ref_str = '-';
        end
        if j <= n_sk
            sk_str = sprintf('%.8f', results{i}.eigs_sk(j));
        else
            sk_str = '-';
        end
        fprintf('  %-12d  %16s  %16s  %16s\n', j, true_str, ref_str, sk_str);
    end
end