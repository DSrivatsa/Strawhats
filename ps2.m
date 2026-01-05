clc;
clear;
close all;

[mixed, Fs] = audioread('mixed.wav');

if size(mixed,2) > 1
    mixed = mean(mixed, 2);
end

mixed = mixed / max(abs(mixed));

winLength = round(0.03 * Fs);
win = hann(winLength, 'periodic');
overlap = round(0.75 * winLength);
nfft = 2^nextpow2(winLength);

[S, f, t] = stft(mixed, Fs, ...
    'Window', win, ...
    'OverlapLength', overlap, ...
    'FFTLength', nfft);

Smag = abs(S);

figure;
imagesc(t, f, 20*log10(Smag + eps));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram of Mixed Signal');
colorbar;

avgEnergy = mean(Smag.^2, 2);
avgEnergy = avgEnergy / max(avgEnergy);

figure;
plot(f, avgEnergy);
xlabel('Frequency (Hz)');
ylabel('Normalized Energy');
title('Average Spectral Energy');
grid on;

[~, idx] = max(avgEnergy);
splitFreq = f(idx);

fprintf('Chosen frequency split: %.2f Hz\n', splitFreq);

M1 = zeros(size(Smag));
M2 = zeros(size(Smag));

for k = 1:length(f)
    if f(k) <= splitFreq
        M1(k,:) = 1;
    else
        M2(k,:) = 1;
    end
end

eps_val = 1e-8;

W1 = (M1 .* Smag).^2 ./ ((M1 .* Smag).^2 + (M2 .* Smag).^2 + eps_val);
W2 = (M2 .* Smag).^2 ./ ((M1 .* Smag).^2 + (M2 .* Smag).^2 + eps_val);

figure;
subplot(121)
imagesc(t, f, W1);
axis xy;
title('Time-Frequency Mask (Source 1)');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

subplot(122)
imagesc(t, f, W2);
axis xy;
title('Time-Frequency Mask (Source 2)');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

S1 = W1 .* S;
S2 = W2 .* S;

src1 = istft(S1, Fs, ...
    'Window', win, ...
    'OverlapLength', overlap, ...
    'FFTLength', nfft);

src2 = istft(S2, Fs, ...
    'Window', win, ...
    'OverlapLength', overlap, ...
    'FFTLength', nfft);

src1 = src1 / max(abs(src1));
src2 = src2 / max(abs(src2));

audiowrite('source1_est.wav', src1, Fs);
audiowrite('source2_est.wav', src2, Fs);

disp('Separated audio files saved.');

figure;
subplot(211)
imagesc(t, f, 20*log10(abs(S1) + eps));
axis xy;
title('Estimated Source 1 Spectrogram');
ylabel('Frequency (Hz)');
colorbar;

subplot(212)
imagesc(t, f, 20*log10(abs(S2) + eps));
axis xy;
title('Estimated Source 2 Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

if exist('clean1.wav','file') && exist('clean2.wav','file')

    [clean1, ~] = audioread('clean1.wav');
    [clean2, ~] = audioread('clean2.wav');

    if size(clean1,2) > 1, clean1 = mean(clean1,2); end
    if size(clean2,2) > 1, clean2 = mean(clean2,2); end

    L = min([length(clean1), length(clean2), length(src1), length(src2)]);
    clean1 = clean1(1:L);
    clean2 = clean2(1:L);
    src1   = src1(1:L);
    src2   = src2(1:L);

    SDR1 = 10*log10(sum(clean1.^2) / sum((clean1 - src1).^2));
    SDR2 = 10*log10(sum(clean2.^2) / sum((clean2 - src2).^2));

    fprintf('\nEvaluation Results:\n');
    fprintf('SDR Source 1: %.2f dB\n', SDR1);
    fprintf('SDR Source 2: %.2f dB\n', SDR2);

else
    disp('Clean reference files not found. Metrics skipped.');
end