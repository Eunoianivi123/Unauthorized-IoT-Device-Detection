"""Fast N-BaIoT feature extractor: O(n) per capture instead of O(n^2).

Drop-in replacement for ``pcap_csv.extract_features_from_pcap`` with identical
semantics: same 115 features, same column order, same eviction rule, same CSV
format.

Why this exists
---------------
``pcap_csv.py`` rebuilds a Python list of every sliding window's values for
every packet (``Window._vals()``), so the work per packet grows with the number
of packets already seen -> O(n^2) over a capture.  This module keeps each
window's values in a numpy buffer and reduces over the live slice, which is
C-speed: the per-packet cost becomes O(n_window) at ~50x lower constant, and
the *dominant* window (L5 = 60 s) stops growing with the capture length for
captures shorter than 60 s.

Verified feature-for-feature against ``pcap_csv.py`` by ``verify_fast_extractor.py``.
"""
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP, Ether

from pcap_csv import (ALL_COLUMNS, BATCH_SIZE, WINDOW_SECS, feature_columns,
                      _flush_batch)

INIT_CAP = 16
COMPACT_AT = 4096
MAX_WINDOW = max(WINDOW_SECS)
STALE_SWEEP_EVERY = 10000


class FastWindow:
    """Time-based sliding window backed by numpy buffers (same eviction rule)."""

    __slots__ = ("duration", "ts", "val", "head", "tail")

    def __init__(self, duration: float):
        self.duration = duration
        self.ts = np.empty(INIT_CAP, dtype=np.float64)
        self.val = np.empty(INIT_CAP, dtype=np.float64)
        self.head = 0
        self.tail = 0

    def _grow(self):
        live = self.tail - self.head
        new_cap = max(self.ts.size * 2, live + 1)
        ts = np.empty(new_cap, dtype=np.float64)
        val = np.empty(new_cap, dtype=np.float64)
        ts[:live] = self.ts[self.head:self.tail]
        val[:live] = self.val[self.head:self.tail]
        self.ts, self.val = ts, val
        self.head, self.tail = 0, live

    def push(self, ts: float, value: float):
        if self.tail >= self.ts.size:
            self._grow()
        self.ts[self.tail] = ts
        self.val[self.tail] = value
        self.tail += 1

        cutoff = ts - self.duration
        ts_a, t = self.ts, self.tail
        h = self.head
        while h < t and ts_a[h] < cutoff:
            h += 1
        self.head = h

        if h >= COMPACT_AT and h * 2 >= t:
            live = t - h
            self.ts[:live] = ts_a[h:t].copy()
            self.val[:live] = self.val[h:t].copy()
            self.head, self.tail = 0, live

    def live(self):
        return self.val[self.head:self.tail]

    def weight(self):
        return float(self.tail - self.head)

    def variance(self):
        v = self.live()
        n = v.size
        if n < 2:
            return 0.0
        mu = float(v.mean())
        d = v - mu
        return float(np.dot(d, d)) / n

    def wmv(self):
        v = self.live()
        n = v.size
        if n == 0:
            return 0.0, 0.0, 0.0
        mu = float(v.mean())
        d = v - mu
        return float(n), mu, float(np.dot(d, d)) / n

    def magnitude(self):
        v = self.live()
        if v.size == 0:
            return 0.0
        return math.sqrt(float(np.dot(v, v)))


class FastScalarStream:
    """5 sliding windows for one scalar stream (e.g. packet sizes)."""

    __slots__ = ("wins",)

    def __init__(self):
        self.wins = [FastWindow(s) for s in WINDOW_SECS]

    def push(self, ts: float, v: float):
        for w in self.wins:
            w.push(ts, v)

    def feats_wmv(self, out, off):
        for i, w in enumerate(self.wins):
            n, mu, var = w.wmv()
            out[off] = n
            out[off + 1] = mu
            out[off + 2] = var
            off += 3


class FastBidirStream:
    """Two scalar streams (fwd/bwd) sharing 5 windows - channel/socket features."""

    __slots__ = ("fwd", "bwd")

    def __init__(self):
        self.fwd = [FastWindow(s) for s in WINDOW_SECS]
        self.bwd = [FastWindow(s) for s in WINDOW_SECS]

    def push_fwd(self, ts: float, v: float):
        for w in self.fwd:
            w.push(ts, v)

    def push_bwd(self, ts: float, v: float):
        for w in self.bwd:
            w.push(ts, v)

    def feats_7(self, out, off):
        for i in range(5):
            fw, bw = self.fwd[i], self.bwd[i]
            fv, bv = fw.live(), bw.live()
            fn, bn = fv.size, bv.size
            weight = float(fn + bn)

            if weight > 0:
                comb = np.concatenate((fv, bv))
                mu_all = float(comb.mean())
                d = comb - mu_all
                var_all = float(np.dot(d, d)) / weight
                std_all = math.sqrt(var_all)
                mag = math.sqrt(float(np.dot(comb, comb)))
            else:
                mu_all = var_all = std_all = mag = 0.0

            var_f = fw.variance()
            var_b = bw.variance()
            radius = math.sqrt(var_f + var_b)

            n_pairs = min(fn, bn)
            if n_pairs:
                cov = float(np.dot(fv[:n_pairs], bv[:n_pairs])) / n_pairs
            else:
                cov = 0.0
            denom = math.sqrt(var_f) * math.sqrt(var_b)
            pcc = cov / denom if denom != 0.0 else 0.0

            out[off] = weight
            out[off + 1] = mu_all
            out[off + 2] = std_all
            out[off + 3] = mag
            out[off + 4] = radius
            out[off + 5] = cov
            out[off + 6] = pcc
            off += 7


class FastNBaIoTExtractor:
    """Stateful, packet-by-packet N-BaIoT feature extractor (same stream mapping
    as ``pcap_csv.NBaIoTExtractor``)."""

    def __init__(self):
        self.mi = defaultdict(FastScalarStream)
        self.h = defaultdict(FastScalarStream)
        self.hh = defaultdict(FastBidirStream)
        self.hh_jit = defaultdict(FastScalarStream)
        self.hh_last = {}
        self.hpHp = defaultdict(FastBidirStream)
        self.hh_last_seen = {}
        self.hpHp_last_seen = {}
        self.n = 0
        self.buf = np.empty(115, dtype=np.float64)

    def _evict_stale(self, now):
        """Drop per-flow streams idle longer than the largest (60 s) window.

        A stream with no packet in >60 s has every window fully evicted, so
        re-creating it on the next packet yields identical features.  The
        ``hh_last`` timestamps are kept (one float per pair) because the jitter
        feature uses the previous packet time even across a 60 s gap.
        """
        cutoff = now - MAX_WINDOW
        stale = [k for k, t in self.hh_last_seen.items() if t < cutoff]
        for k in stale:
            del self.hh_last_seen[k]
            self.hh.pop(k, None)
            self.hh_jit.pop(k, None)
        stale = [k for k, t in self.hpHp_last_seen.items() if t < cutoff]
        for k in stale:
            del self.hpHp_last_seen[k]
            self.hpHp.pop(k, None)

    @staticmethod
    def _canon(a, b):
        return (a, b) if a <= b else (b, a)

    def process(self, pkt):
        if not pkt.haslayer(IP):
            return None

        ts = float(pkt.time)
        size = float(len(pkt))
        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        src_mac = pkt[Ether].src if pkt.haslayer(Ether) else "00:00:00:00:00:00"

        sport, dport = 0, 0
        if pkt.haslayer(TCP):
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif pkt.haslayer(UDP):
            sport, dport = pkt[UDP].sport, pkt[UDP].dport

        mi_key = src_ip
        h_key = (src_mac, src_ip)
        hh_key = self._canon(src_ip, dst_ip)
        hpHp_key = self._canon((src_ip, sport), (dst_ip, dport))
        hh_fwd = (src_ip <= dst_ip)
        hpHp_fwd = ((src_ip, sport) <= (dst_ip, dport))

        self.n += 1
        if self.n % STALE_SWEEP_EVERY == 0:
            self._evict_stale(ts)
        self.hh_last_seen[hh_key] = ts
        self.hpHp_last_seen[hpHp_key] = ts

        self.mi[mi_key].push(ts, size)
        self.h[h_key].push(ts, size)

        if hh_fwd:
            self.hh[hh_key].push_fwd(ts, size)
        else:
            self.hh[hh_key].push_bwd(ts, size)

        if hh_key in self.hh_last:
            self.hh_jit[hh_key].push(ts, ts - self.hh_last[hh_key])
        self.hh_last[hh_key] = ts

        if hpHp_fwd:
            self.hpHp[hpHp_key].push_fwd(ts, size)
        else:
            self.hpHp[hpHp_key].push_bwd(ts, size)

        buf = self.buf
        self.mi[mi_key].feats_wmv(buf, 0)
        self.h[h_key].feats_wmv(buf, 15)
        self.hh[hh_key].feats_7(buf, 30)
        self.hh_jit[hh_key].feats_wmv(buf, 65)
        self.hpHp[hpHp_key].feats_7(buf, 80)

        return [ts, src_ip, dst_ip] + buf.tolist()


def extract_features_to_csv(pcap_path: str, output_csv: str,
                            batch_size: int = BATCH_SIZE) -> int:
    """Stream N-BaIoT features straight to CSV in O(1) row memory.

    Same 115 features / column order / eviction rule as
    ``extract_features_from_pcap``, but never keeps the full capture in RAM.
    Returns the number of feature rows written.

    Use this for long captures (hundreds of thousands of packets): the
    DataFrame-returning variant accumulates every row and OOMs around ~200k
    rows on the 3.8 GB pipeline container.
    """
    print(f"[*] Streaming: {pcap_path} (CSV-only, no in-memory rows)")

    extractor = FastNBaIoTExtractor()
    batch = []
    total = 0
    skipped = 0
    written = 0
    first_chunk = True

    def flush() -> None:
        nonlocal first_chunk, written
        df_chunk = pd.DataFrame(batch, columns=ALL_COLUMNS)
        df_chunk.to_csv(output_csv, mode="w" if first_chunk else "a",
                        header=first_chunk, index=False)
        written += len(batch)
        batch.clear()
        first_chunk = False
        print(f"    flushed {written} rows ...")

    with PcapReader(pcap_path) as reader:
        for i, pkt in enumerate(reader):
            row = extractor.process(pkt)
            if row is None:
                skipped += 1
            else:
                batch.append(row)
                total += 1

            if len(batch) >= batch_size:
                flush()

            if (i + 1) % 5000 == 0:
                print(f"    {i + 1} packets processed ...")

    if batch:
        flush()

    print(f"[*] Rows: {total}  |  Non-IP skipped: {skipped}")
    print(f"[*] Saved -> {output_csv}")
    return total


def extract_features_from_pcap(pcap_path: str, output_csv: str = None,
                               batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """Same contract as ``pcap_csv.extract_features_from_pcap``."""
    print(f"[*] Streaming: {pcap_path}")

    extractor = FastNBaIoTExtractor()
    all_rows = []
    batch = []
    total = 0
    skipped = 0
    first_chunk = True

    with PcapReader(pcap_path) as reader:
        for i, pkt in enumerate(reader):
            row = extractor.process(pkt)
            if row is None:
                skipped += 1
            else:
                batch.append(row)
                total += 1

            if output_csv and len(batch) >= batch_size:
                first_chunk = _flush_batch(batch, all_rows, first_chunk, output_csv)

            if (i + 1) % 5000 == 0:
                print(f"    {i + 1} packets processed ...")

    if batch:
        if output_csv:
            _flush_batch(batch, all_rows, first_chunk, output_csv)
        else:
            all_rows.extend(batch)

    print(f"[*] Rows: {total}  |  Non-IP skipped: {skipped}")

    df = pd.DataFrame(all_rows, columns=ALL_COLUMNS)

    if output_csv:
        print(f"[*] Saved -> {output_csv}")

    return df


if __name__ == "__main__":
    pcap_file = sys.argv[1] if len(sys.argv) > 1 else 'attack_capture.pcap'
    output_file = pcap_file.rsplit(".", 1)[0] + "_features.csv"
    df = extract_features_from_pcap(pcap_file, output_csv=output_file)
    print(f"\n{'=' * 52}")
    print(f"  Shape          : {df.shape}")
    print(f"  Unique src IPs : {df['src_ip'].nunique()}")
    print(f"  Time range     : {df['timestamp'].min():.3f} -> {df['timestamp'].max():.3f} s")
    print(f"{'=' * 52}")
    print(f"[OK] Done.  Output: {output_file}")
