"""
琉球音階（レ・ラ抜き）を使った波打ち際向けのリール用BGM生成
Ryukyu scale: C, E, F, G, B (no D, no A)
"""

import numpy as np
import wave
import struct

SAMPLE_RATE = 44100
DURATION = 62  # ~1分2秒

# 琉球音階の基音周波数（Hz）- C4基準
# ド=C4, ミ=E4, ファ=F4, ソ=G4, シ=B4, ド=C5, ミ=E5...
RYUKYU_FREQS = {
    'C3': 130.81, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00, 'B3': 246.94,
    'C4': 261.63, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'B4': 493.88,
    'C5': 523.25, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99, 'B5': 987.77,
    'C6': 1046.50,
}

def note(name, octave_shift=0):
    f = RYUKYU_FREQS[name]
    return f * (2 ** octave_shift)

def sine_wave(freq, duration, sr=SAMPLE_RATE, amplitude=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def warm_tone(freq, duration, sr=SAMPLE_RATE, amplitude=0.5):
    """基音＋倍音で温かみのある音色"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = (amplitude * 0.6 * np.sin(2 * np.pi * freq * t) +
            amplitude * 0.25 * np.sin(2 * np.pi * freq * 2 * t) +
            amplitude * 0.10 * np.sin(2 * np.pi * freq * 3 * t) +
            amplitude * 0.05 * np.sin(2 * np.pi * freq * 4 * t))
    return wave

def adsr_envelope(samples, attack=0.12, decay=0.15, sustain=0.65, release=0.25, sr=SAMPLE_RATE):
    n = len(samples)
    env = np.ones(n)
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s_level = sustain

    env[:a] = np.linspace(0, 1, a)
    env[a:a+d] = np.linspace(1, s_level, d)
    env[a+d:n-r] = s_level
    if n - r > 0:
        env[n-r:] = np.linspace(s_level, 0, r)
    return samples * env

def reverb(signal, delay_sec=0.06, decay=0.35, sr=SAMPLE_RATE):
    delay_samples = int(delay_sec * sr)
    out = signal.copy()
    for i in range(1, 5):
        delayed = np.zeros_like(signal)
        d = delay_samples * i
        if d < len(signal):
            delayed[d:] = signal[:-d] * (decay ** i)
        out += delayed
    return out * 0.7

def ocean_pad(duration, sr=SAMPLE_RATE):
    """波のような低周波パッド"""
    n = int(duration * sr)
    # ゆっくり揺れるノイズ
    noise = np.random.randn(n) * 0.015
    # 低域フィルタ的な効果（移動平均）
    kernel = np.ones(2000) / 2000
    filtered = np.convolve(noise, kernel, mode='same')
    # 波のリズムで揺らす
    t = np.linspace(0, duration, n)
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.08 * t) * np.sin(2 * np.pi * 0.13 * t)
    return filtered * swell

def bass_drone(freq, duration, sr=SAMPLE_RATE, amp=0.12):
    """ゆったりしたベースドローン"""
    t = np.linspace(0, duration, int(sr * duration))
    lfo = 0.95 + 0.05 * np.sin(2 * np.pi * 0.2 * t)
    d = (amp * 0.7 * np.sin(2 * np.pi * freq * t) +
         amp * 0.3 * np.sin(2 * np.pi * freq * 2 * t)) * lfo
    return d

def place_note(buffer, freq, start_sec, dur_sec, amplitude=0.28, sr=SAMPLE_RATE):
    samples = warm_tone(freq, dur_sec, sr, amplitude)
    samples = adsr_envelope(samples, attack=0.1, decay=0.2, sustain=0.6, release=0.3)
    start = int(start_sec * sr)
    end = start + len(samples)
    if end > len(buffer):
        samples = samples[:len(buffer) - start]
    buffer[start:start+len(samples)] += samples

def normalize(signal, peak=0.82):
    m = np.max(np.abs(signal))
    if m > 0:
        signal = signal / m * peak
    return signal

def save_wav(filename, signal, sr=SAMPLE_RATE):
    signal = normalize(signal)
    signal_int = (signal * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal_int.tobytes())
    print(f"Saved: {filename}")


# ============================================================
# Track 1: 波打ち際の朝 - ゆったりした単音メロディ
# テンポ: 52 BPM / 穏やか・透明感
# ============================================================
def make_track1():
    sr = SAMPLE_RATE
    buf = np.zeros(int(DURATION * sr))

    # 海のパッド
    buf += ocean_pad(DURATION, sr) * 1.2
    # ベースドローン: C3
    buf += bass_drone(RYUKYU_FREQS['C3'], DURATION, sr, 0.10)
    buf += bass_drone(RYUKYU_FREQS['G3'], DURATION, sr, 0.06)

    # メロディ（琉球音階: C E F G B）
    # フレーズ1 (0-15s)
    mel = [
        ('G4', 0.0,  3.0), ('E4', 3.5,  2.5), ('F4', 6.5,  2.0),
        ('G4', 9.0,  2.5), ('E4', 12.0, 2.0), ('C4', 14.5, 2.5),
    ]
    # フレーズ2 (17-32s)
    mel += [
        ('B4', 17.5, 2.5), ('G4', 20.5, 2.0), ('F4', 23.0, 2.5),
        ('E4', 26.0, 2.0), ('G4', 28.5, 3.5),
    ]
    # フレーズ3 (33-48s)
    mel += [
        ('C5', 33.0, 2.5), ('B4', 36.0, 2.0), ('G4', 38.5, 2.5),
        ('F4', 41.5, 2.0), ('E4', 44.0, 2.5), ('C4', 47.0, 4.0),
    ]
    # フレーズ4 (52-62s)
    mel += [
        ('G4', 52.0, 2.5), ('F4', 55.0, 2.0), ('E4', 57.5, 2.0), ('C4', 60.0, 3.0),
    ]

    for name, start, dur in mel:
        place_note(buf, RYUKYU_FREQS[name], start, dur, amplitude=0.28, sr=sr)

    buf = reverb(buf, delay_sec=0.07, decay=0.3)
    save_wav('track1_naminoasa.wav', buf, sr)


# ============================================================
# Track 2: アダンの森の風 - 少し動きのあるアルペジオ感
# テンポ: 60 BPM / 木漏れ日・葉擦れ
# ============================================================
def make_track2():
    sr = SAMPLE_RATE
    buf = np.zeros(int(DURATION * sr))

    buf += ocean_pad(DURATION, sr) * 0.9
    buf += bass_drone(RYUKYU_FREQS['C3'], DURATION, sr, 0.09)
    buf += bass_drone(RYUKYU_FREQS['E3'], DURATION, sr, 0.05)

    # アルペジオ的なパターン（4秒サイクル）
    arp_pattern = [
        ('C4', 0.0, 1.6), ('E4', 1.0, 1.6), ('G4', 2.0, 1.6), ('B4', 3.0, 1.8),
    ]
    arp_pattern2 = [
        ('G4', 0.0, 1.8), ('E4', 1.0, 1.6), ('C4', 2.0, 1.6), ('F4', 3.0, 1.8),
    ]
    arp_pattern3 = [
        ('F4', 0.0, 1.6), ('G4', 1.0, 1.6), ('B4', 2.0, 1.6), ('C5', 3.0, 2.0),
    ]

    cycles = [arp_pattern, arp_pattern2, arp_pattern, arp_pattern3,
              arp_pattern2, arp_pattern, arp_pattern3, arp_pattern2,
              arp_pattern, arp_pattern2, arp_pattern, arp_pattern3,
              arp_pattern, arp_pattern2, arp_pattern3]
    offset = 0.0
    for cycle in cycles:
        for name, rel, dur in cycle:
            t = offset + rel
            if t < DURATION:
                place_note(buf, RYUKYU_FREQS[name], t, dur, amplitude=0.20, sr=sr)
        offset += 4.0
        if offset >= DURATION:
            break

    # メロディ重ね
    top_mel = [
        ('G5', 2.0, 3.5), ('F5', 6.0, 3.0), ('E5', 10.0, 3.5),
        ('G5', 14.0, 3.0), ('B4', 18.0, 4.0),
        ('C5', 24.0, 3.0), ('G4', 28.0, 3.5), ('F4', 32.0, 3.0),
        ('E5', 37.0, 3.5), ('G5', 42.0, 3.0), ('F5', 47.0, 3.5),
        ('E5', 52.0, 3.0), ('C5', 57.0, 5.0),
    ]
    for name, start, dur in top_mel:
        place_note(buf, RYUKYU_FREQS[name], start, dur, amplitude=0.22, sr=sr)

    buf = reverb(buf, delay_sec=0.09, decay=0.32)
    save_wav('track2_mori_no_kaze.wav', buf, sr)


# ============================================================
# Track 3: 父の収穫 - 力強さと温かみ
# テンポ: 68 BPM / 少しリズミカルだが穏やか
# ============================================================
def make_track3():
    sr = SAMPLE_RATE
    buf = np.zeros(int(DURATION * sr))

    buf += ocean_pad(DURATION, sr) * 0.7
    buf += bass_drone(RYUKYU_FREQS['C3'], DURATION, sr, 0.11)
    buf += bass_drone(RYUKYU_FREQS['G3'], DURATION, sr, 0.07)
    buf += bass_drone(RYUKYU_FREQS['E3'], DURATION, sr, 0.05)

    # 低めの力強いメロディ
    mel = [
        # intro
        ('C4', 0.0, 2.8), ('E4', 3.0, 2.5),
        # A
        ('G4', 6.0, 2.5), ('F4', 9.0, 2.0), ('E4', 11.5, 2.5),
        ('G4', 14.5, 2.0), ('B4', 17.0, 3.0),
        # B
        ('C5', 21.0, 2.5), ('B4', 24.0, 2.0), ('G4', 26.5, 2.5),
        ('F4', 29.5, 2.0), ('E4', 32.0, 2.5),
        # C
        ('G4', 35.0, 2.0), ('E4', 37.5, 2.5), ('C4', 40.5, 2.5),
        ('E4', 43.5, 2.0), ('F4', 46.0, 2.5), ('G4', 48.5, 3.0),
        # outro
        ('E4', 52.5, 2.5), ('C4', 55.5, 2.0), ('G3', 58.5, 4.5),
    ]
    for name, start, dur in mel:
        place_note(buf, RYUKYU_FREQS[name], start, dur, amplitude=0.32, sr=sr)

    # 副旋律（低め）
    sub = [
        ('C3', 6.0, 5.5), ('G3', 14.0, 6.0), ('F3', 21.0, 5.0),
        ('E3', 29.0, 6.0), ('C3', 38.0, 6.0), ('G3', 48.0, 5.5),
        ('C3', 55.0, 8.0),
    ]
    for name, start, dur in sub:
        place_note(buf, RYUKYU_FREQS[name], start, dur, amplitude=0.16, sr=sr)

    buf = reverb(buf, delay_sec=0.08, decay=0.28)
    save_wav('track3_chichi_no_shukaku.wav', buf, sr)


# ============================================================
# Track 4: アダンピアス - 海と手仕事の静けさ
# テンポ: 50 BPM / 最もゆったり・瞑想的
# ============================================================
def make_track4():
    sr = SAMPLE_RATE
    buf = np.zeros(int(DURATION * sr))

    buf += ocean_pad(DURATION, sr) * 1.4
    buf += bass_drone(RYUKYU_FREQS['C3'], DURATION, sr, 0.12)
    buf += bass_drone(RYUKYU_FREQS['G3'], DURATION, sr, 0.07)
    buf += bass_drone(RYUKYU_FREQS['C4'], DURATION, sr, 0.04)

    # 長い音価で瞑想的に
    mel = [
        ('E4', 1.0,  5.0), ('C4', 7.0,  4.5), ('G4', 13.0, 5.5),
        ('F4', 20.0, 5.0), ('E4', 27.0, 5.0), ('B4', 34.0, 5.5),
        ('G4', 41.0, 5.0), ('F4', 47.0, 4.5), ('C4', 53.0, 5.0),
        ('E4', 59.0, 4.0),
    ]
    for name, start, dur in mel:
        place_note(buf, RYUKYU_FREQS[name], start, dur, amplitude=0.26, sr=sr)

    # 高音の点描
    sparkle = [
        ('C5', 4.0, 2.5), ('B4', 11.0, 2.5), ('G5', 18.0, 2.5),
        ('F5', 25.0, 2.5), ('E5', 33.0, 2.5), ('C5', 40.0, 2.5),
        ('G5', 48.0, 2.5), ('B4', 56.0, 2.5),
    ]
    for name, start, dur in sparkle:
        place_note(buf, RYUKYU_FREQS[name], start, dur, amplitude=0.14, sr=sr)

    buf = reverb(buf, delay_sec=0.10, decay=0.38)
    save_wav('track4_adan_pierce.wav', buf, sr)


if __name__ == '__main__':
    print("琉球音階BGM生成中...")
    make_track1()
    make_track2()
    make_track3()
    make_track4()
    print("\n完成！4つの音源を生成しました。")
    print("  track1_naminoasa.wav      - 波打ち際の朝（52BPM・透明感）")
    print("  track2_mori_no_kaze.wav   - アダンの森の風（60BPM・アルペジオ）")
    print("  track3_chichi_no_shukaku.wav - 父の収穫（68BPM・力強さ）")
    print("  track4_adan_pierce.wav    - アダンピアス（50BPM・瞑想的）")
