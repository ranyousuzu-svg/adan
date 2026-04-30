"""
琉球音階 三線BGM 改良版
- Karplus-Strong: ハニング窓励振 + 微小ランダム減衰（有機的な揺らぎ）
- 複数弦のデチューンで自然なコーラス感
- タイミング/強さのヒューマナイズ
- 不規則な反射時間の自然なリバーブ（尾部ローパス）
- 各曲 28 秒（15-30秒素材として使いやすい長さ）
- テンポ 62-68 BPM（アダンの森・父の収穫相当）
"""

import numpy as np
import wave
from scipy.signal import butter, sosfilt

SR = 44100
DUR = 28

# 琉球音階（レ・ラ抜き） C E F G B
def hz(note, oct):
    s = {'C':0,'E':4,'F':5,'G':7,'B':11}
    return 440.0 * 2**((s[note] + (oct-4)*12 - 9)/12)

N = {f'{n}{o}': hz(n,o) for n in 'CEFGB' for o in range(2,7)}


# ── 音合成 ───────────────────────────────────────────────

def ks(freq, dur, amp, damping):
    """Karplus-Strong: ハニング窓励振 + 微小ランダム減衰"""
    delay = max(2, int(round(SR / freq)))
    n = int(SR * dur)
    buf = np.zeros(delay)
    ex = min(delay, max(2, int(delay * 0.55)))
    buf[:ex] = np.random.randn(ex) * np.hanning(ex) * amp
    dv = np.random.randn(n) * 6e-5          # 減衰の微小変動（有機感）
    out = np.zeros(n)
    idx = 0
    for i in range(n):
        out[i] = buf[idx]
        nxt = (idx + 1) % delay
        buf[idx] = (damping + dv[i]) * 0.5 * (buf[idx] + buf[nxt])
        idx = nxt
    return out

def sanshin(freq, dur, amp=0.70):
    """3弦わずかにデチューン → 自然なコーラス・うなり"""
    tail = dur + 0.7
    s1 = ks(freq,          tail, amp*0.65, 0.9974)
    s2 = ks(freq * 1.0022, tail, amp*0.23, 0.9969)
    s3 = ks(freq * 0.9990, tail, amp*0.12, 0.9978)
    # 攻撃部に短いクリック（バチで弾く感触）
    ck_n = int(0.003 * SR)
    click = np.random.randn(ck_n) * amp * 0.18 * np.exp(-np.linspace(0,18,ck_n))
    combined = s1 + s2 + s3
    combined[:ck_n] += click
    return combined

def place(buf, freq, t0, dur, amp=0.68):
    """タイミング・強さにヒューマナイズ"""
    t0 = max(0.0, t0 + np.random.uniform(-0.020, 0.020))
    amp = amp * np.random.uniform(0.93, 1.07)
    note = sanshin(freq, dur, amp)
    s = int(t0 * SR)
    e = min(s + len(note), len(buf))
    if s < len(buf):
        buf[s:e] += note[:e-s]


# ── 背景 ─────────────────────────────────────────────────

def ocean(dur):
    n = int(dur * SR)
    raw = np.random.randn(n) * 0.011
    filt = np.convolve(raw, np.ones(5500)/5500, mode='same')
    t = np.linspace(0, dur, n)
    return filt * (0.60 + 0.40 * np.sin(2*np.pi*0.060*t) * np.sin(2*np.pi*0.088*t))

def drone(freq, dur, amp=0.07):
    t = np.linspace(0, dur, int(dur*SR))
    return amp*(np.sin(2*np.pi*freq*t)+0.30*np.sin(2*np.pi*freq*2*t)) \
           *(1+0.018*np.sin(2*np.pi*0.10*t))


# ── エフェクト ────────────────────────────────────────────

def natural_reverb(sig):
    """不規則な反射タップ + 尾部ローパスで有機的な空間感"""
    sos = butter(3, 4200/(SR/2), btype='low', output='sos')
    taps = [
        (0.021,0.44),(0.039,0.36),(0.063,0.28),
        (0.094,0.22),(0.138,0.17),(0.195,0.12),
        (0.271,0.08),(0.376,0.05),(0.501,0.03),
    ]
    out = sig.copy()
    for ds, g in taps:
        d = int(ds * SR)
        if d < len(sig):
            tail = np.zeros_like(sig)
            tail[d:] = sig[:-d] * g
            out += sosfilt(sos, tail)
    return out * 0.60

def save(fname, sig):
    fade = int(1.8 * SR)
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig /= (np.max(np.abs(sig)) + 1e-9)
    sig *= 0.86
    with wave.open(fname, 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((sig * 32767).astype(np.int16).tobytes())
    print(f'Saved: {fname}')


# ── Track 1: 波打ち際の朝  62 BPM ─────────────────────────
# G→E→F→G→E→C  空白  B→G→E→F→G→C  ゆったり下降解決
def track1():
    buf = np.zeros(int(DUR * SR))
    buf += ocean(DUR) * 1.30
    buf += drone(N['C3'], DUR, 0.085)
    buf += drone(N['G3'], DUR, 0.045)

    b = 60/62   # 0.968 s/beat
    mel = [
        # フレーズA：G→E→F→G→E→C（長い解決）
        (N['G4'], 0.0*b,   2.0*b, 0.72),
        (N['E4'], 2.0*b,   1.5*b, 0.67),
        (N['F4'], 3.5*b,   1.0*b, 0.63),
        (N['G4'], 4.5*b,   1.5*b, 0.70),
        (N['E4'], 6.0*b,   2.0*b, 0.68),
        (N['C4'], 8.0*b,   4.0*b, 0.74),   # ← 長い解決
        # フレーズB：B→G→E→F→G→C
        (N['B4'], 13.5*b,  2.0*b, 0.68),
        (N['G4'], 15.5*b,  1.5*b, 0.65),
        (N['E4'], 17.0*b,  1.0*b, 0.63),
        (N['F4'], 18.0*b,  0.8*b, 0.60),
        (N['G4'], 18.8*b,  1.5*b, 0.68),
        (N['E4'], 20.3*b,  1.2*b, 0.65),
        (N['C4'], 21.5*b,  5.5*b, 0.74),   # ← 締め・長い余韻
    ]
    for f, t, d, a in mel:
        if t < DUR: place(buf, f, t, min(d, DUR-t), a)
    save('track1_naminoasa.wav', natural_reverb(buf))


# ── Track 2: アダンの森  64 BPM ──────────────────────────
# 流れるように連なるメロディ・少し動きが多い
def track2():
    buf = np.zeros(int(DUR * SR))
    buf += ocean(DUR) * 0.90
    buf += drone(N['C3'], DUR, 0.075)
    buf += drone(N['E3'], DUR, 0.040)

    b = 60/64   # 0.9375 s/beat
    mel = [
        # フレーズA：上へ広がる
        (N['E4'], 0.0*b,  1.5*b, 0.65),
        (N['G4'], 1.5*b,  1.0*b, 0.68),
        (N['B4'], 2.5*b,  2.0*b, 0.72),
        (N['G4'], 4.5*b,  1.0*b, 0.65),
        (N['F4'], 5.5*b,  0.7*b, 0.60),
        (N['E4'], 6.2*b,  1.5*b, 0.67),
        (N['C4'], 7.7*b,  2.8*b, 0.72),
        # フレーズB：高音域へ
        (N['G4'], 11.2*b, 1.0*b, 0.65),
        (N['B4'], 12.2*b, 1.5*b, 0.70),
        (N['C5'], 13.7*b, 2.0*b, 0.67),
        (N['B4'], 15.7*b, 1.0*b, 0.63),
        (N['G4'], 16.7*b, 1.0*b, 0.65),
        (N['F4'], 17.7*b, 0.7*b, 0.60),
        (N['E4'], 18.4*b, 1.0*b, 0.65),
        (N['G4'], 19.4*b, 1.5*b, 0.68),
        (N['C4'], 20.9*b, 5.0*b, 0.72),
    ]
    for f, t, d, a in mel:
        if t < DUR: place(buf, f, t, min(d, DUR-t), a)
    save('track2_mori_no_kaze.wav', natural_reverb(buf))


# ── Track 3: 父の収穫  68 BPM ─────────────────────────────
# 力強く弾む・テンポ感あり
def track3():
    buf = np.zeros(int(DUR * SR))
    buf += ocean(DUR) * 0.65
    buf += drone(N['C3'], DUR, 0.090)
    buf += drone(N['G3'], DUR, 0.055)

    b = 60/68   # 0.882 s/beat
    mel = [
        # 力強いイントロ
        (N['G4'], 0.0*b,  1.0*b, 0.80),
        (N['C4'], 1.0*b,  1.5*b, 0.78),
        (N['E4'], 2.5*b,  0.6*b, 0.72),
        (N['G4'], 3.1*b,  1.5*b, 0.80),
        # フレーズA
        (N['B4'], 5.0*b,  1.5*b, 0.76),
        (N['G4'], 6.5*b,  0.6*b, 0.70),
        (N['F4'], 7.1*b,  0.8*b, 0.67),
        (N['E4'], 7.9*b,  1.0*b, 0.72),
        (N['G4'], 8.9*b,  1.5*b, 0.76),
        (N['C4'], 10.4*b, 2.5*b, 0.80),
        # フレーズB
        (N['E4'], 13.5*b, 0.6*b, 0.70),
        (N['G4'], 14.1*b, 1.0*b, 0.75),
        (N['B4'], 15.1*b, 1.5*b, 0.78),
        (N['C5'], 16.6*b, 2.0*b, 0.72),
        (N['B4'], 18.6*b, 0.6*b, 0.67),
        (N['G4'], 19.2*b, 0.8*b, 0.70),
        (N['F4'], 20.0*b, 0.6*b, 0.65),
        (N['E4'], 20.6*b, 1.0*b, 0.70),
        (N['C4'], 21.6*b, 5.5*b, 0.80),
    ]
    for f, t, d, a in mel:
        if t < DUR: place(buf, f, t, min(d, DUR-t), a)
    save('track3_chichi_no_shukaku.wav', natural_reverb(buf))


# ── Track 4: アダンピアス  60 BPM ────────────────────────
# 最もゆったり・沈黙を大切に
def track4():
    buf = np.zeros(int(DUR * SR))
    buf += ocean(DUR) * 1.50
    buf += drone(N['C3'], DUR, 0.100)
    buf += drone(N['G3'], DUR, 0.055)
    buf += drone(N['C2'], DUR, 0.045)

    b = 60/60   # 1.0 s/beat
    mel = [
        (N['E4'], 1.0*b,  3.0*b, 0.66),
        (N['C4'], 5.5*b,  3.5*b, 0.72),
        # 沈黙 2拍
        (N['G4'], 11.5*b, 2.5*b, 0.68),
        (N['B4'], 14.5*b, 3.0*b, 0.65),
        # 沈黙 1.5拍
        (N['G4'], 19.5*b, 2.0*b, 0.65),
        (N['F4'], 22.0*b, 1.5*b, 0.60),
        (N['E4'], 24.0*b, 1.5*b, 0.65),
        (N['C4'], 26.0*b, 5.0*b, 0.72),
    ]
    for f, t, d, a in mel:
        if t < DUR: place(buf, f, t, min(d, DUR-t), a)
    save('track4_adan_pierce.wav', natural_reverb(buf))


if __name__ == '__main__':
    np.random.seed(7)
    print('生成中...')
    track1(); track2(); track3(); track4()
    print('\n完成（各28秒）')
