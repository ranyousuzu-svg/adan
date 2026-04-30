"""
琉球音階（レ・ラ抜き）三線ベースBGM
Karplus-Strong合成で弦をはじく音を再現
"""

import numpy as np
import wave

SR = 44100
DURATION = 63

# 琉球音階の周波数（C, E, F, G, B）レとラを含まない
def hz(note, octave):
    semis = {'C': 0, 'E': 4, 'F': 5, 'G': 7, 'B': 11}
    return 440.0 * 2 ** ((semis[note] + (octave - 4) * 12 - 9) / 12)

N = {}
for n in ['C','E','F','G','B']:
    for o in range(2, 7):
        N[f'{n}{o}'] = hz(n, o)


def ks(freq, dur, amp=0.75, damping=0.997):
    """Karplus-Strong プラックドストリング合成"""
    delay = max(2, int(round(SR / freq)))
    n = int(SR * dur)
    buf = np.zeros(delay)
    # 励振：短いノイズバースト（弦をはじく）
    excite = min(delay, max(2, int(delay * 0.5)))
    buf[:excite] = np.random.randn(excite) * amp
    out = np.zeros(n)
    idx = 0
    for i in range(n):
        out[i] = buf[idx]
        nxt = (idx + 1) % delay
        buf[idx] = damping * 0.5 * (buf[idx] + buf[nxt])
        idx = nxt
    return out


def sanshin(freq, dur, amp=0.72):
    """三線らしい音：KS + 明るい倍音（サワリ感）"""
    base = ks(freq, dur + 0.4, amp * 0.82, damping=0.9975)
    t = np.linspace(0, dur + 0.4, len(base))
    # 高速減衰する倍音でサワリ感（三線特有の明るさ）
    buzz_decay = np.exp(-12 * t)
    buzz = amp * 0.18 * np.sin(2 * np.pi * freq * 2 * t) * buzz_decay
    buzz += amp * 0.07 * np.sin(2 * np.pi * freq * 3 * t) * buzz_decay
    return base + buzz


def place(buf, freq, start_s, dur_s, amp=0.68):
    note = sanshin(freq, dur_s, amp)
    s = int(start_s * SR)
    e = min(s + len(note), len(buf))
    buf[s:e] += note[:e - s]


def ocean(dur):
    """波のアンビエントパッド"""
    n = int(dur * SR)
    noise = np.random.randn(n) * 0.014
    kernel = np.ones(4000) / 4000
    filtered = np.convolve(noise, kernel, mode='same')
    t = np.linspace(0, dur, n)
    swell = 0.65 + 0.35 * np.sin(2 * np.pi * 0.06 * t) * np.sin(2 * np.pi * 0.09 * t)
    return filtered * swell


def bass_drone(freq, dur, amp=0.10):
    """低音ドローン（背景の支え）"""
    t = np.linspace(0, dur, int(dur * SR))
    lfo = 1 + 0.03 * np.sin(2 * np.pi * 0.15 * t)
    wave = amp * np.sin(2 * np.pi * freq * t) * lfo
    wave += amp * 0.4 * np.sin(2 * np.pi * freq * 2 * t) * lfo
    return wave


def reverb(sig, ms=75, decay=0.32):
    d = int(ms * SR / 1000)
    out = sig.copy()
    for i in range(1, 5):
        dd = d * i
        if dd < len(sig):
            delayed = np.zeros_like(sig)
            delayed[dd:] = sig[:-dd] * (decay ** i)
            out += delayed
    return out * 0.70


def save(fname, sig):
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.88
    with wave.open(fname, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((sig * 32767).astype(np.int16).tobytes())
    print(f'Saved: {fname}')


# ============================================================
# Track 1: 波打ち際の朝  52BPM  ゆったり単線メロディ
# ============================================================
def track1():
    buf = np.zeros(int(DURATION * SR))
    buf += ocean(DURATION) * 1.3
    buf += bass_drone(N['C3'], DURATION, 0.09)
    buf += bass_drone(N['G3'], DURATION, 0.05)

    b = 60 / 52  # 1拍 ≈ 1.15s

    mel = [
        # フレーズA：Gから下りてCへ解決
        (N['G4'], 0*b,    2*b,  0.70),
        (N['E4'], 2*b,    1.5*b,0.65),
        (N['F4'], 3.5*b,  1*b,  0.62),
        (N['G4'], 4.5*b,  1.5*b,0.68),
        (N['E4'], 6*b,    2*b,  0.65),
        (N['C4'], 8*b,    3.5*b,0.72),  # 解決・長め

        # フレーズB：上へ広がる
        (N['E4'], 12*b,   1*b,  0.63),
        (N['G4'], 13*b,   1.5*b,0.68),
        (N['B4'], 14.5*b, 2.5*b,0.72),
        (N['G4'], 17*b,   1.5*b,0.65),
        (N['F4'], 18.5*b, 1*b,  0.60),
        (N['E4'], 19.5*b, 1.5*b,0.65),
        (N['G4'], 21*b,   1*b,  0.62),
        (N['C4'], 22*b,   4*b,  0.72),  # 長い解決

        # フレーズC：少し変化
        (N['G4'], 27*b,   1.5*b,0.68),
        (N['B4'], 28.5*b, 2*b,  0.70),
        (N['C5'], 30.5*b, 2.5*b,0.65),
        (N['B4'], 33*b,   1.5*b,0.62),
        (N['G4'], 34.5*b, 1.5*b,0.65),
        (N['F4'], 36*b,   1*b,  0.60),
        (N['E4'], 37*b,   1.5*b,0.65),
        (N['C4'], 38.5*b, 3.5*b,0.72),

        # フレーズD（締め）
        (N['G4'], 43*b,   1.5*b,0.68),
        (N['E4'], 44.5*b, 1*b,  0.62),
        (N['F4'], 45.5*b, 1.5*b,0.63),
        (N['G4'], 47*b,   2*b,  0.68),
        (N['E4'], 49*b,   1.5*b,0.65),
        (N['C4'], 50.5*b, 5*b,  0.72),  # 長い終止
    ]
    for (f, st, dur, amp) in mel:
        if st < DURATION:
            place(buf, f, st, min(dur, DURATION - st), amp)

    buf = reverb(buf, ms=80, decay=0.30)
    save('track1_naminoasa.wav', buf)


# ============================================================
# Track 2: アダンの森  60BPM  流れるアルペジオ＋メロディ
# ============================================================
def track2():
    buf = np.zeros(int(DURATION * SR))
    buf += ocean(DURATION) * 0.9
    buf += bass_drone(N['C3'], DURATION, 0.08)
    buf += bass_drone(N['E3'], DURATION, 0.04)

    b = 60 / 60  # 1拍 = 1.0s

    # 低音アルペジオ（背景）
    arp = [
        (N['C3'], 0.50), (N['E3'], 0.50), (N['G3'], 0.50), (N['C4'], 0.50),
    ]
    t = 0.0
    while t < DURATION - 2:
        for (f, dur) in arp:
            place(buf, f, t, dur, amp=0.30)
            t += dur
        t += 0.0  # 休みなし・流れるように

    # 上声メロディ（流れる旋律）
    mel = [
        # フレーズA
        (N['E4'], 0*b,    1.5*b, 0.65),
        (N['G4'], 1.5*b,  1*b,   0.68),
        (N['B4'], 2.5*b,  2*b,   0.72),
        (N['G4'], 4.5*b,  1*b,   0.65),
        (N['F4'], 5.5*b,  0.5*b, 0.60),
        (N['E4'], 6*b,    2*b,   0.67),
        (N['C4'], 8*b,    2.5*b, 0.70),

        # フレーズB（上へ）
        (N['G4'], 11*b,   1*b,   0.65),
        (N['B4'], 12*b,   1.5*b, 0.70),
        (N['C5'], 13.5*b, 2.5*b, 0.68),
        (N['B4'], 16*b,   1*b,   0.62),
        (N['G4'], 17*b,   1*b,   0.65),
        (N['E4'], 18*b,   1.5*b, 0.65),
        (N['F4'], 19.5*b, 0.5*b, 0.58),
        (N['G4'], 20*b,   2.5*b, 0.68),
        (N['C4'], 22.5*b, 2.5*b, 0.72),

        # フレーズC（高音域）
        (N['E5'], 26*b,   1.5*b, 0.60),
        (N['G5'], 27.5*b, 2*b,   0.55),
        (N['F5'], 29.5*b, 1*b,   0.58),
        (N['E5'], 30.5*b, 1.5*b, 0.60),
        (N['C5'], 32*b,   2*b,   0.65),
        (N['B4'], 34*b,   1.5*b, 0.63),
        (N['G4'], 35.5*b, 1*b,   0.65),
        (N['F4'], 36.5*b, 0.5*b, 0.58),
        (N['E4'], 37*b,   2*b,   0.65),
        (N['C4'], 39*b,   2.5*b, 0.70),

        # フレーズD（締め）
        (N['G4'], 42.5*b, 1.5*b, 0.68),
        (N['B4'], 44*b,   2*b,   0.70),
        (N['G4'], 46*b,   1*b,   0.63),
        (N['E4'], 47*b,   1*b,   0.63),
        (N['F4'], 48*b,   1*b,   0.60),
        (N['G4'], 49*b,   1.5*b, 0.65),
        (N['E4'], 50.5*b, 1*b,   0.62),
        (N['C4'], 51.5*b, 5*b,   0.70),
    ]
    for (f, st, dur, amp) in mel:
        if st < DURATION:
            place(buf, f, st, min(dur, DURATION - st), amp)

    buf = reverb(buf, ms=70, decay=0.28)
    save('track2_mori_no_kaze.wav', buf)


# ============================================================
# Track 3: 父の収穫  68BPM  力強く弾む
# ============================================================
def track3():
    buf = np.zeros(int(DURATION * SR))
    buf += ocean(DURATION) * 0.65
    buf += bass_drone(N['C3'], DURATION, 0.10)
    buf += bass_drone(N['G3'], DURATION, 0.06)

    b = 60 / 68  # 1拍 ≈ 0.88s

    mel = [
        # 力強いイントロ
        (N['G4'], 0*b,    1*b,   0.80),
        (N['C4'], 1*b,    1.5*b, 0.78),
        (N['E4'], 2.5*b,  0.5*b, 0.72),
        (N['G4'], 3*b,    1.5*b, 0.80),

        # フレーズA
        (N['B4'], 5*b,    1.5*b, 0.75),
        (N['G4'], 6.5*b,  0.5*b, 0.70),
        (N['F4'], 7*b,    1*b,   0.68),
        (N['E4'], 8*b,    1*b,   0.72),
        (N['G4'], 9*b,    1.5*b, 0.75),
        (N['C4'], 10.5*b, 2.5*b, 0.78),

        # フレーズB
        (N['E4'], 14*b,   0.5*b, 0.70),
        (N['G4'], 14.5*b, 1*b,   0.75),
        (N['B4'], 15.5*b, 1.5*b, 0.78),
        (N['C5'], 17*b,   2*b,   0.72),
        (N['B4'], 19*b,   0.5*b, 0.68),
        (N['G4'], 19.5*b, 1*b,   0.72),
        (N['F4'], 20.5*b, 0.5*b, 0.65),
        (N['E4'], 21*b,   1.5*b, 0.72),
        (N['C4'], 22.5*b, 2.5*b, 0.78),

        # フレーズC（反復・力強く）
        (N['G4'], 26*b,   0.5*b, 0.80),
        (N['E4'], 26.5*b, 0.5*b, 0.75),
        (N['G4'], 27*b,   1*b,   0.80),
        (N['B4'], 28*b,   1.5*b, 0.78),
        (N['G4'], 29.5*b, 0.5*b, 0.72),
        (N['F4'], 30*b,   1*b,   0.68),
        (N['G4'], 31*b,   1.5*b, 0.75),
        (N['E4'], 32.5*b, 1*b,   0.70),
        (N['C4'], 33.5*b, 2.5*b, 0.78),

        # フレーズD
        (N['C5'], 37*b,   1.5*b, 0.70),
        (N['B4'], 38.5*b, 0.5*b, 0.65),
        (N['G4'], 39*b,   1*b,   0.72),
        (N['F4'], 40*b,   0.5*b, 0.65),
        (N['E4'], 40.5*b, 1*b,   0.70),
        (N['G4'], 41.5*b, 1.5*b, 0.75),
        (N['C4'], 43*b,   2.5*b, 0.80),

        # 締め
        (N['G4'], 46.5*b, 1*b,   0.78),
        (N['E4'], 47.5*b, 0.5*b, 0.72),
        (N['F4'], 48*b,   0.5*b, 0.68),
        (N['G4'], 48.5*b, 1*b,   0.75),
        (N['C4'], 49.5*b, 5*b,   0.80),
    ]
    for (f, st, dur, amp) in mel:
        if st < DURATION:
            place(buf, f, st, min(dur, DURATION - st), amp)

    buf = reverb(buf, ms=60, decay=0.25)
    save('track3_chichi_no_shukaku.wav', buf)


# ============================================================
# Track 4: アダンピアス  50BPM  最もゆったり・瞑想的
# ============================================================
def track4():
    buf = np.zeros(int(DURATION * SR))
    buf += ocean(DURATION) * 1.5
    buf += bass_drone(N['C3'], DURATION, 0.11)
    buf += bass_drone(N['G3'], DURATION, 0.06)
    buf += bass_drone(N['C2'], DURATION, 0.05)

    b = 60 / 50  # 1拍 = 1.2s

    mel = [
        # 息の長い単音・沈黙を大事に
        (N['E4'], 1*b,    3*b,   0.65),
        (N['C4'], 5*b,    3.5*b, 0.70),
        # 沈黙 2拍

        (N['G4'], 11*b,   2.5*b, 0.68),
        (N['B4'], 14*b,   3*b,   0.65),
        # 沈黙 1.5拍

        (N['G4'], 19.5*b, 2*b,   0.65),
        (N['F4'], 22*b,   1.5*b, 0.60),
        (N['E4'], 24*b,   3.5*b, 0.67),
        # 沈黙 1拍

        (N['C4'], 29.5*b, 4*b,   0.72),  # 長い解決
        # 沈黙 1.5拍

        (N['B4'], 36*b,   2.5*b, 0.63),
        (N['G4'], 39*b,   2*b,   0.65),
        (N['F4'], 41.5*b, 1.5*b, 0.60),
        (N['E4'], 43.5*b, 3*b,   0.67),
        # 沈黙 1拍

        (N['C4'], 48*b,   6*b,   0.72),  # 最後の長い余韻
    ]
    for (f, st, dur, amp) in mel:
        if st < DURATION:
            place(buf, f, st, min(dur, DURATION - st), amp)

    buf = reverb(buf, ms=100, decay=0.38)
    save('track4_adan_pierce.wav', buf)


if __name__ == '__main__':
    np.random.seed(42)
    print('三線音源生成中...')
    track1()
    track2()
    track3()
    track4()
    print('\n完成！')
    print('  track1_naminoasa.wav      - 波打ち際の朝（52BPM）')
    print('  track2_mori_no_kaze.wav   - アダンの森（60BPM）')
    print('  track3_chichi_no_shukaku.wav - 父の収穫（68BPM）')
    print('  track4_adan_pierce.wav    - アダンピアス（50BPM）')
