clc;
clear;
close all;

Fs = 16000;
frameLen = 256;
filterLen = 512;
w = zeros(filterLen,1);
x_buf = zeros(filterLen,1);
mu = 0.6;
eps_val = 1e-6;
gamma = 0.7;

disp('Acoustic echo cancellation simulation');
disp('Generating synthetic echo scenario');

duration = 5;
t = (0:1/Fs:duration-1/Fs)';

far_end = 0.5 * sin(2*pi*300*t) .* (1 + 0.3*sin(2*pi*3*t));
far_end = far_end + 0.3*randn(size(t));
near_end = 0.3 * sin(2*pi*500*t) .* (1 + 0.5*sin(2*pi*2*t));
near_end(1:Fs*2) = 0;
near_end(Fs*2:end) = near_end(Fs*2:end) * 0.5;
echo_path = [1; 0.5; 0.3; 0.2; 0.1; zeros(filterLen-5,1)];
echo_signal = filter(echo_path, 1, far_end);
mic_signal = near_end + 0.7*echo_signal + 0.05*randn(size(t));
numFrames = floor(length(mic_signal) / frameLen);
output = zeros(size(mic_signal));
echo_estimate = zeros(size(mic_signal));
fprintf('Processing %d frames...\n', numFrames);

for frame = 1:numFrames
    idx = (frame-1)*frameLen + 1 : frame*frameLen;
    d = mic_signal(idx);
    x = far_end(idx);
    y_hat = zeros(frameLen,1);
    e = zeros(frameLen,1);
    for n = 1:frameLen
        x_buf = [x(n); x_buf(1:end-1)];
        y_hat(n) = w' * x_buf;
        e(n) = d(n) - y_hat(n);
        if abs(d(n)) / (max(abs(x_buf)) + eps_val) < gamma
            w = w + (mu * e(n) * x_buf) / (x_buf' * x_buf + eps_val);
        end
    end
    output(idx) = e;
    echo_estimate(idx) = y_hat;
end

echo_reduction = 10*log10(mean(mic_signal.^2) / mean(output.^2));
figure('Position', [100 100 1200 800]);

subplot(411)
plot(t, far_end);
title('Far-End Signal (Reference)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(412)
plot(t, mic_signal);
title('Microphone Signal (Near-End + Echo + Noise)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(413)
plot(t, echo_estimate);
title('Estimated Echo');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(414)
plot(t, output);
title('Echo Cancelled Output');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

figure('Position', [100 100 1200 600]);
subplot(211)
spectrogram(mic_signal, hann(256), 192, 512, Fs, 'yaxis');
title('Spectrogram: Microphone Signal (with Echo)');
colorbar;

subplot(212)
spectrogram(output, hann(256), 192, 512, Fs, 'yaxis');
title('Spectrogram: Echo Cancelled Output');
colorbar;

figure;
plot(echo_path(1:50), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(w(1:50), 'r--x', 'LineWidth', 2, 'MarkerSize', 6);
legend('True Echo Path', 'Estimated Filter', 'Location', 'best');
title('Filter Convergence - Estimated vs True Echo Path');
xlabel('Tap Index');
ylabel('Filter Coefficient');
grid on;

erle = zeros(numFrames, 1);
for frame = 1:numFrames
    idx = (frame-1)*frameLen + 1 : frame*frameLen;
    erle(frame) = 10*log10(mean(mic_signal(idx).^2) / (mean(output(idx).^2) + eps_val));
end

figure;
plot((1:numFrames)*frameLen/Fs, erle, 'LineWidth', 2);
title('Echo Return Loss Enhancement (ERLE) Over Time');
xlabel('Time (s)');
ylabel('ERLE (dB)');
grid on;
ylim([0 max(erle)+5]);

fprintf('\n=== RESULTS ===\n');
fprintf('Echo Reduction: %.2f dB\n', echo_reduction);
fprintf('Filter Length: %d taps\n', filterLen);
fprintf('Step Size (mu): %.2f\n', mu);
fprintf('Double-talk threshold (gamma): %.2f\n', gamma);
fprintf('Mean ERLE: %.2f dB\n', mean(erle(erle>0)));

audiowrite('mic_with_echo.wav', mic_signal/max(abs(mic_signal)), Fs);
audiowrite('echo_cancelled.wav', output/max(abs(output)), Fs);
disp('Audio files saved: mic_with_echo.wav, echo_cancelled.wav');