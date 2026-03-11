# fuzzy_sd.py
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np


def _collect_mfs_from_section(sec: dict, nmf: int) -> list:
    mfs = []
    for j in range(1, nmf + 1):
        key = f"MF{j}"
        if key not in sec:
            raise KeyError(f"Missing {key} in section")
        mfs.append(FIS._parse_mf(sec[key]))
    return mfs

# -----------------------------
# Membership functions
# -----------------------------
def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.maximum(
        0.0,
        np.minimum((x - a) / (b - a + 1e-12), (c - x) / (c - b + 1e-12)),
    )

def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a + 1e-12)
    right = (d - x) / (d - c + 1e-12)
    return np.maximum(0.0, np.minimum(np.minimum(left, 1.0), right))

def gaussmf(x: np.ndarray, sigma: float, c: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - c) / (sigma + 1e-12)) ** 2)

def eval_mf(x: np.ndarray, mf_type: str, params: List[float]) -> np.ndarray:
    t = mf_type.lower()
    if t == "trimf":
        return trimf(x, params[0], params[1], params[2])
    if t == "trapmf":
        return trapmf(x, params[0], params[1], params[2], params[3])
    if t == "gaussmf":
        return gaussmf(x, params[0], params[1])
    raise NotImplementedError(f"MF type not supported: {mf_type}")


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class MF:
    name: str
    mf_type: str
    params: List[float]

@dataclass
class Var:
    name: str
    vmin: float
    vmax: float
    mfs: List[MF]

@dataclass
class Rule:
    in_idx: List[int]      # indices for each input (1-based), 0 means "don't care"
    out_idx: int           # output MF index (1-based)
    connective: int        # 1=AND, 2=OR
    weight: float


class FIS:
    """
    Mamdani FIS evaluator for the common .fis format you pasted:
      "i1 i2 i3 i4, o (conn) : w"

    Methods supported:
      AndMethod: min / prod
      OrMethod:  max / probor
      ImpMethod: min / prod
      AggMethod: max / sum
      DefuzzMethod: centroid
    """

    def __init__(
        self,
        and_method: str,
        or_method: str,
        imp_method: str,
        agg_method: str,
        defuzz_method: str,
        inputs: List[Var],
        output: Var,
        rules: List[Rule],
        grid_n: int = 501,
    ):
        self.and_method = and_method.lower()
        self.or_method = or_method.lower()
        self.imp_method = imp_method.lower()
        self.agg_method = agg_method.lower()
        self.defuzz_method = defuzz_method.lower()
        self.inputs = inputs
        self.output = output
        self.rules = rules

        if self.defuzz_method != "centroid":
            raise NotImplementedError(f"Only centroid supported now, got: {defuzz_method}")

        self._y = np.linspace(output.vmin, output.vmax, int(grid_n))
        self._out_mf_vals = [eval_mf(self._y, mf.mf_type, mf.params) for mf in output.mfs]

    # -----------------------------
    # Parsing helpers
    # -----------------------------
    @staticmethod
    def _parse_ini(path: str) -> Dict[str, Dict[str, str]]:
        sec = None
        data: Dict[str, Dict[str, str]] = {}
        rules_lines: List[str] = []

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue

                m = re.match(r"\[(.+?)\]", line)
                if m:
                    # при выходе из [Rules] — сохраняем накопленные правила
                    if sec == "Rules" and rules_lines:
                        data.setdefault("Rules", {})["Rules"] = "\n".join(rules_lines)
                        rules_lines = []

                    sec = m.group(1)
                    data.setdefault(sec, {})
                    continue

                if sec is None:
                    continue

                # Особый случай: [Rules] — строки без "="
                if sec == "Rules":
                    # правила идут как plain text
                    rules_lines.append(line)
                    continue

                # обычные секции: key=value
                if "=" in line:
                    k, v = line.split("=", 1)
                    data[sec][k.strip()] = v.strip()

        # если файл закончился внутри [Rules]
        if sec == "Rules" and rules_lines:
            data.setdefault("Rules", {})["Rules"] = "\n".join(rules_lines)

        return data

    @staticmethod
    def _parse_range(s: str) -> Tuple[float, float]:
        s = s.strip().strip("[]")
        a, b = s.split()
        return float(a), float(b)

    @staticmethod
    def _parse_mf(mf_line: str) -> MF:
        # "'name':'trapmf',[1 2 3 4]"
        m = re.match(r"'(.+?)'\s*:\s*'(.+?)'\s*,\s*\[(.+)\]", mf_line)
        if not m:
            raise ValueError(f"Bad MF line: {mf_line}")
        name, mf_type, params_s = m.group(1), m.group(2), m.group(3)
        params = [float(x) for x in re.split(r"[\s,]+", params_s.strip()) if x]
        return MF(name=name, mf_type=mf_type, params=params)

    @staticmethod
    def _parse_rules_block(rules_text: str, n_in: int) -> List[Rule]:
        """
        Your format:
          "1 1 1 1, 2 (1) : 1"

        Meaning:
          inputs: 4 ints
          output: 1 int
          connective: (1) AND, (2) OR
          weight: after ':'
        """
        rules: List[Rule] = []
        lines = [ln.strip() for ln in rules_text.splitlines() if ln.strip()]

        # regex for your exact format
        # group1: inputs, group2: output, group3: connective, group4: weight
        pat = re.compile(r"^(.+?),\s*(\d+)\s*\((\d+)\)\s*:\s*([0-9.]+)\s*$")

        for ln in lines:
            ln = re.sub(r"\s+", " ", ln)
            m = pat.match(ln)
            if not m:
                raise ValueError(f"Unsupported rule format: {ln}")

            in_part = m.group(1).strip()
            out_idx = int(m.group(2))
            conn = int(m.group(3))
            w = float(m.group(4))

            ins = [int(x) for x in in_part.split()]
            if len(ins) != n_in:
                raise ValueError(f"Rule size mismatch (expected {n_in} inputs): {ln}")

            rules.append(Rule(in_idx=ins, out_idx=out_idx, connective=conn, weight=w))

        return rules

    @classmethod
    @classmethod
    def from_fis(cls, path: str, grid_n: int = 501) -> "FIS":
        d = cls._parse_ini(path)
        sys = d["System"]

        n_in = int(sys["NumInputs"])
        n_out = int(sys["NumOutputs"])
        if n_out != 1:
            raise NotImplementedError("This implementation supports exactly 1 output.")

        fis_type = sys.get("Type", "").strip("'").lower()
        if fis_type != "mamdani":
            raise NotImplementedError(f"Only mamdani supported, got: {fis_type}")

        and_m = sys.get("AndMethod", "").strip("'")
        or_m = sys.get("OrMethod", "").strip("'")
        imp_m = sys.get("ImpMethod", "").strip("'")
        agg_m = sys.get("AggMethod", "").strip("'")
        defz = sys.get("DefuzzMethod", "").strip("'")

        inputs: List[Var] = []
        for i in range(1, n_in + 1):
            sec = d[f"Input{i}"]
            name = sec["Name"].strip("'")
            vmin, vmax = cls._parse_range(sec["Range"])
            nmf = int(sec["NumMFs"])

            # MF прямо внутри [Inputi] как MF1=..., MF2=...
            mfs = []
            for j in range(1, nmf + 1):
                k = f"MF{j}"
                if k not in sec:
                    raise KeyError(f"Input{i}: missing {k}")
                mfs.append(cls._parse_mf(sec[k]))

            inputs.append(Var(name=name, vmin=vmin, vmax=vmax, mfs=mfs))

        out_sec = d["Output1"]
        out_name = out_sec["Name"].strip("'")
        out_vmin, out_vmax = cls._parse_range(out_sec["Range"])
        out_nmf = int(out_sec["NumMFs"])

        out_mfs = []
        for j in range(1, out_nmf + 1):
            k = f"MF{j}"
            if k not in out_sec:
                raise KeyError(f"Output1: missing {k}")
            out_mfs.append(cls._parse_mf(out_sec[k]))

        output = Var(name=out_name, vmin=out_vmin, vmax=out_vmax, mfs=out_mfs)

        rules_text = d.get("Rules", {}).get("Rules", "")
        rules = cls._parse_rules_block(rules_text, n_in=n_in)

        return cls(
            and_method=and_m,
            or_method=or_m,
            imp_method=imp_m,
            agg_method=agg_m,
            defuzz_method=defz,
            inputs=inputs,
            output=output,
            rules=rules,
            grid_n=grid_n,
        )

    # -----------------------------
    # Operators
    # -----------------------------
    def _t_norm(self, a: float, b: float) -> float:
        if self.and_method == "min":
            return min(a, b)
        if self.and_method == "prod":
            return a * b
        raise NotImplementedError(f"AndMethod not supported: {self.and_method}")

    def _s_norm(self, a: float, b: float) -> float:
        if self.or_method == "max":
            return max(a, b)
        if self.or_method == "probor":
            return a + b - a * b
        raise NotImplementedError(f"OrMethod not supported: {self.or_method}")

    def _implicate(self, alpha: float, mf_vals: np.ndarray) -> np.ndarray:
        if self.imp_method == "min":
            return np.minimum(alpha, mf_vals)
        if self.imp_method == "prod":
            return alpha * mf_vals
        raise NotImplementedError(f"ImpMethod not supported: {self.imp_method}")

    def _aggregate(self, agg: np.ndarray, new: np.ndarray) -> np.ndarray:
        if self.agg_method == "max":
            return np.maximum(agg, new)
        if self.agg_method == "sum":
            return np.clip(agg + new, 0.0, 1.0)
        raise NotImplementedError(f"AggMethod not supported: {self.agg_method}")

    def _defuzz_centroid(self, y: np.ndarray, mu: np.ndarray) -> float:
        num = float(np.trapz(y * mu, y))
        den = float(np.trapz(mu, y))
        if den < 1e-12:
            return float(0.5 * (y[0] + y[-1]))
        return num / den

    # -----------------------------
    # Evaluation
    # -----------------------------
    def eval(self, x: List[float]) -> float:
        if len(x) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(x)}")

        # fuzzify point
        degs: List[List[float]] = []
        for val, var in zip(x, self.inputs):
            v = float(val)
            mvals = []
            for mf in var.mfs:
                mu = float(eval_mf(np.array([v]), mf.mf_type, mf.params)[0])
                mvals.append(mu)
            degs.append(mvals)

        agg_mu = np.zeros_like(self._y)

        for rule in self.rules:
            alpha: Optional[float] = None

            for i_in, mf_idx in enumerate(rule.in_idx):
                if mf_idx == 0:
                    continue
                mu = degs[i_in][mf_idx - 1]
                if alpha is None:
                    alpha = mu
                else:
                    alpha = (
                        self._t_norm(alpha, mu)
                        if rule.connective == 1
                        else self._s_norm(alpha, mu)
                    )

            if alpha is None:
                alpha = 0.0

            alpha *= rule.weight

            if rule.out_idx == 0:
                continue

            base = self._out_mf_vals[rule.out_idx - 1]
            clipped = self._implicate(alpha, base)
            agg_mu = self._aggregate(agg_mu, clipped)

        return float(self._defuzz_centroid(self._y, agg_mu))