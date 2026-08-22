"""PCAP -> 115-feature N-BaIoT CSV converter (streaming, low memory)."""
import math
import sys
from collections import defaultdict, deque

try:
    from scapy.all import PcapReader, IP, TCP, UDP, Ether
except ImportError:
    print("ERROR: scapy not installed.  Run: pip install scapy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed.  Run: pip install pandas")
    sys.exit(1)


# ── Temporal windows (label, duration) ────────────────────
WINDOWS = [
    ("L5",    60.0),
    ("L3",    10.0),
    ("L1",     1.5),
    ("L0.1",   0.5),
    ("L0.01",  0.1),
]
WINDOW_SECS = [s for _, s in WINDOWS]

# ── 115 feature column names (UCI dataset convention) ─────
feature_columns = [
    "MI_dir_L5_weight","MI_dir_L5_mean","MI_dir_L5_variance",
    "MI_dir_L3_weight","MI_dir_L3_mean","MI_dir_L3_variance",
    "MI_dir_L1_weight","MI_dir_L1_mean","MI_dir_L1_variance",
    "MI_dir_L0.1_weight","MI_dir_L0.1_mean","MI_dir_L0.1_variance",
    "MI_dir_L0.01_weight","MI_dir_L0.01_mean","MI_dir_L0.01_variance",
    "H_L5_weight","H_L5_mean","H_L5_variance",
    "H_L3_weight","H_L3_mean","H_L3_variance",
    "H_L1_weight","H_L1_mean","H_L1_variance",
    "H_L0.1_weight","H_L0.1_mean","H_L0.1_variance",
    "H_L0.01_weight","H_L0.01_mean","H_L0.01_variance",
    "HH_L5_weight","HH_L5_mean","HH_L5_std","HH_L5_magnitude","HH_L5_radius","HH_L5_covariance","HH_L5_pcc",
    "HH_L3_weight","HH_L3_mean","HH_L3_std","HH_L3_magnitude","HH_L3_radius","HH_L3_covariance","HH_L3_pcc",
    "HH_L1_weight","HH_L1_mean","HH_L1_std","HH_L1_magnitude","HH_L1_radius","HH_L1_covariance","HH_L1_pcc",
    "HH_L0.1_weight","HH_L0.1_mean","HH_L0.1_std","HH_L0.1_magnitude","HH_L0.1_radius","HH_L0.1_covariance","HH_L0.1_pcc",
    "HH_L0.01_weight","HH_L0.01_mean","HH_L0.01_std","HH_L0.01_magnitude","HH_L0.01_radius","HH_L0.01_covariance","HH_L0.01_pcc",
    "HH_jit_L5_weight","HH_jit_L5_mean","HH_jit_L5_variance",
    "HH_jit_L3_weight","HH_jit_L3_mean","HH_jit_L3_variance",
    "HH_jit_L1_weight","HH_jit_L1_mean","HH_jit_L1_variance",
    "HH_jit_L0.1_weight","HH_jit_L0.1_mean","HH_jit_L0.1_variance",
    "HH_jit_L0.01_weight","HH_jit_L0.01_mean","HH_jit_L0.01_variance",
    "HpHp_L5_weight","HpHp_L5_mean","HpHp_L5_std","HpHp_L5_magnitude","HpHp_L5_radius","HpHp_L5_covariance","HpHp_L5_pcc",
    "HpHp_L3_weight","HpHp_L3_mean","HpHp_L3_std","HpHp_L3_magnitude","HpHp_L3_radius","HpHp_L3_covariance","HpHp_L3_pcc",
    "HpHp_L1_weight","HpHp_L1_mean","HpHp_L1_std","HpHp_L1_magnitude","HpHp_L1_radius","HpHp_L1_covariance","HpHp_L1_pcc",
    "HpHp_L0.1_weight","HpHp_L0.1_mean","HpHp_L0.1_std","HpHp_L0.1_magnitude","HpHp_L0.1_radius","HpHp_L0.1_covariance","HpHp_L0.1_pcc",
    "HpHp_L0.01_weight","HpHp_L0.01_mean","HpHp_L0.01_std","HpHp_L0.01_magnitude","HpHp_L0.01_radius","HpHp_L0.01_covariance","HpHp_L0.01_pcc",
]

assert len(feature_columns) == 115, f"BUG: expected 115, got {len(feature_columns)}"

# CSV columns including metadata
ALL_COLUMNS = ["timestamp", "src_ip", "dst_ip"] + feature_columns

# Rows to accumulate before flushing to disk
BATCH_SIZE = 10_000


# ── Sliding-window accumulator ────────────────────────────
class Window:
    """Time-based sliding window of (timestamp, scalar) pairs."""

    __slots__ = ("duration", "buf")

    def __init__(self, duration: float):
        self.duration = duration
        self.buf: deque = deque()   # (ts, value)

    def push(self, ts: float, value: float):
        self.buf.append((ts, value))
        cutoff = ts - self.duration
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()

    def weight(self) -> float:
        return float(len(self.buf))

    def _vals(self):
        return [v for _, v in self.buf]

    @staticmethod
    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def _variance(vals):
        n = len(vals)
        if n < 2:
            return 0.0
        mu = sum(vals) / n
        return sum((x - mu) ** 2 for x in vals) / n

    def wmv(self):
        """Return (weight, mean, variance) without building vals twice."""
        vals = self._vals()
        return float(len(vals)), self._mean(vals), self._variance(vals)

    def std(self) -> float:
        return math.sqrt(self._variance(self._vals()))

    def magnitude(self) -> float:
        """L2 norm = sqrt(sum of squares)."""
        return math.sqrt(sum(v * v for _, v in self.buf)) if self.buf else 0.0


# ── Per-stream helpers ────────────────────────────────────
class ScalarStream:
    """5 sliding windows for a single scalar stream (e.g. packet sizes)."""

    __slots__ = ("wins",)

    def __init__(self):
        self.wins = [Window(s) for s in WINDOW_SECS]

    def push(self, ts: float, v: float):
        for w in self.wins:
            w.push(ts, v)

    def feats_wmv(self) -> list:
        """[weight, mean, variance] x 5 = 15 values."""
        out = []
        for w in self.wins:
            out += w.wmv()
        return out


class BidirStream:
    """Two scalar streams (fwd/bwd) sharing 5 windows - channel/socket features."""

    __slots__ = ("fwd", "bwd")

    def __init__(self):
        self.fwd = [Window(s) for s in WINDOW_SECS]
        self.bwd = [Window(s) for s in WINDOW_SECS]

    def push_fwd(self, ts: float, v: float):
        for w in self.fwd:
            w.push(ts, v)

    def push_bwd(self, ts: float, v: float):
        for w in self.bwd:
            w.push(ts, v)

    def feats_7(self) -> list:
        """[weight, mean, std, magnitude, radius, covariance, pcc] x 5 = 35 values."""
        out = []
        sqrt = math.sqrt
        for i in range(5):
            fv = self.fwd[i]._vals()
            bv = self.bwd[i]._vals()
            fn, bn = len(fv), len(bv)

            weight = float(fn + bn)
            combined = fv + bv
            n_all = fn + bn
            mu_all = sum(combined) / n_all if n_all else 0.0
            var_all = sum((x - mu_all) ** 2 for x in combined) / n_all if n_all >= 2 else 0.0
            std_all = sqrt(var_all)
            mag = sqrt(sum(v * v for v in combined)) if combined else 0.0

            # Per-direction variance -> radius
            mu_f = sum(fv) / fn if fn else 0.0
            mu_b = sum(bv) / bn if bn else 0.0
            var_f = sum((x - mu_f) ** 2 for x in fv) / fn if fn >= 2 else 0.0
            var_b = sum((x - mu_b) ** 2 for x in bv) / bn if bn >= 2 else 0.0
            radius = sqrt(var_f + var_b)

            # Matched-pair covariance & Pearson correlation
            n_pairs = min(fn, bn)
            cov = sum(fv[j] * bv[j] for j in range(n_pairs)) / n_pairs if n_pairs else 0.0
            denom = sqrt(var_f) * sqrt(var_b)
            pcc = cov / denom if denom != 0.0 else 0.0

            out += [weight, mu_all, std_all, mag, radius, cov, pcc]
        return out


# ── Main packet extractor ─────────────────────────────────
class NBaIoTExtractor:
    """Stateful, packet-by-packet N-BaIoT feature extractor.

    Stream mapping (paper -> implementation):
      MI     src IP              -> ScalarStream keyed by src_ip
      H      src MAC-IP          -> ScalarStream keyed by (mac, src_ip)
      HH     channel src<->dst   -> BidirStream  keyed by canonical (src_ip, dst_ip)
      HH_jit channel jitter      -> ScalarStream keyed by canonical (src_ip, dst_ip)
      HpHp   socket src:p<->dst:p-> BidirStream  keyed by canonical socket pair
    """

    def __init__(self):
        self.mi = defaultdict(ScalarStream)
        self.h = defaultdict(ScalarStream)
        self.hh = defaultdict(BidirStream)
        self.hh_jit = defaultdict(ScalarStream)
        self.hh_last = {}
        self.hpHp = defaultdict(BidirStream)

    @staticmethod
    def _canon(a, b):
        return (a, b) if a <= b else (b, a)

    def process(self, pkt) -> list | None:
        """Return [ts, src_ip, dst_ip] + 115 features, or None if no IP layer."""
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

        # ── Stream keys & direction flags ───────────────────
        mi_key = src_ip
        h_key = (src_mac, src_ip)
        hh_key = self._canon(src_ip, dst_ip)
        hpHp_key = self._canon((src_ip, sport), (dst_ip, dport))
        hh_fwd = (src_ip <= dst_ip)
        hpHp_fwd = ((src_ip, sport) <= (dst_ip, dport))

        # ── Update each accumulator ─────────────────────────
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

        # ── Build flat row: MI 15 + H 15 + HH 35 + jitter 15 + HpHp 35 ──
        feats = (
            self.mi[mi_key].feats_wmv() +
            self.h[h_key].feats_wmv() +
            self.hh[hh_key].feats_7() +
            self.hh_jit[hh_key].feats_wmv() +
            self.hpHp[hpHp_key].feats_7()
        )

        return [ts, src_ip, dst_ip] + feats


# ── Public API ────────────────────────────────────────────
def _flush_batch(batch, all_rows, first_chunk, output_csv):
    """Append a batch to the CSV file and memory. Returns the new first_chunk flag."""
    df_chunk = pd.DataFrame(batch, columns=ALL_COLUMNS)
    df_chunk.to_csv(output_csv, mode="w" if first_chunk else "a",
                    header=first_chunk, index=False)
    all_rows.extend(batch)
    batch.clear()
    print(f"    flushed {len(all_rows)} rows ...")
    return False


def extract_features_from_pcap(pcap_path: str, output_csv: str = None,
                               batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """Extract N-BaIoT features from a pcap file.

    Uses streaming PcapReader + batched CSV writes to keep memory low.
    Returns a DataFrame with columns: timestamp, src_ip, dst_ip + 115 features.
    """
    print(f"[*] Streaming: {pcap_path}")

    extractor = NBaIoTExtractor()
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

            # Flush to disk when the batch fills up
            if output_csv and len(batch) >= batch_size:
                first_chunk = _flush_batch(batch, all_rows, first_chunk, output_csv)

            if (i + 1) % 5000 == 0:
                print(f"    {i + 1} packets processed ...")

    # Final flush
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


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    pcap_file = 'attack_capture.pcap'
    output_file = pcap_file.rsplit(".", 1)[0] + "_features.csv"

    df = extract_features_from_pcap(pcap_file, output_csv=output_file)

    print(f"\n{'=' * 52}")
    print(f"  Shape          : {df.shape}")
    print(f"  Unique src IPs : {df['src_ip'].nunique()}")
    print(f"  Time range     : {df['timestamp'].min():.3f} -> {df['timestamp'].max():.3f} s")
    print(f"{'=' * 52}")
    print(f"[OK] Done.  Output: {output_file}")
