#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, shutil, tempfile, zipfile
from pathlib import Path
from typing import Any
import arviz as az

SIM_RE = re.compile(r"sim(\d+)\.nc$")

def dims_to_dict(obj: Any) -> dict[str, int]:
    return {str(k): int(v) for k, v in obj.sizes.items()}

def inspect_member(zf: zipfile.ZipFile, member: str, tmpdir: Path) -> dict[str, Any]:
    target = tmpdir / Path(member).name
    with zf.open(member) as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16*1024*1024)
    idata = az.from_netcdf(target)
    groups = list(idata.groups())
    result = {
        "member": member,
        "groups": groups,
        "has_posterior": "posterior" in groups,
        "has_posterior_predictive": "posterior_predictive" in groups,
        "has_sample_stats": "sample_stats" in groups,
        "has_observed_data": "observed_data" in groups,
    }
    if "posterior" in groups:
        result["posterior_dims"] = dims_to_dict(idata.posterior)
        result["posterior_variables"] = sorted(map(str, idata.posterior.data_vars))
    if "posterior_predictive" in groups:
        result["posterior_predictive_dims"] = dims_to_dict(idata.posterior_predictive)
        result["posterior_predictive_variables"] = sorted(map(str, idata.posterior_predictive.data_vars))
    if "sample_stats" in groups:
        result["sample_stats_dims"] = dims_to_dict(idata.sample_stats)
        result["sample_stats_variables"] = sorted(map(str, idata.sample_stats.data_vars))
    target.unlink(missing_ok=True)
    return result

def inspect_archive(path: Path, sample_count: int) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        members = sorted(n for n in zf.namelist() if n.lower().endswith(".nc"))
        if not members:
            raise RuntimeError(f"Nenhum NetCDF em {path}")
        ids = sorted(int(m.group(1)) for n in members if (m := SIM_RE.search(Path(n).name)))
        idx = sorted(set([0, len(members)//2, len(members)-1]))[:max(1, sample_count)]
        with tempfile.TemporaryDirectory(prefix="nano_trace_inspect_") as td:
            samples = [inspect_member(zf, members[i], Path(td)) for i in idx]
    return {
        "archive": str(path), "size_bytes": path.stat().st_size,
        "netcdf_count": len(members),
        "simulation_ids_min": min(ids), "simulation_ids_max": max(ids),
        "missing_simulation_ids_0_149": sorted(set(range(150))-set(ids)),
        "sampled_files": samples,
    }

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--mass-zip", type=Path, required=True)
    ap.add_argument("--pvs-zip", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-count", type=int, default=3)
    a=ap.parse_args()
    report={"mass":inspect_archive(a.mass_zip,a.sample_count),"pvs":inspect_archive(a.pvs_zip,a.sample_count)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    samples=report["mass"]["sampled_files"]+report["pvs"]["sampled_files"]
    ok=all(x["has_posterior_predictive"] and "y_obs" in x.get("posterior_predictive_variables",[]) for x in samples)
    print("\nDECISÃO:")
    if ok:
        print("[OK] posterior_predictive/y_obs confirmado. A sensibilidade de domínio pode ser recalculada sem novo MCMC.")
        return 0
    print("[ATENÇÃO] posterior_predictive/y_obs não confirmado em todos os traces amostrados.")
    return 2
if __name__=="__main__": raise SystemExit(main())
