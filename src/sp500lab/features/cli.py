"""`sp500lab features ...` - build, inspect and audit the feature layer.

    sp500lab features build            compute and cache the panel
    sp500lab features list             every feature, with its coverage
    sp500lab features check            the leakage test - run this before trusting any
                                       backtest that used features
    sp500lab features show mom_12_1    one feature's cross-section over time

`check` is the one that matters. Everything else is convenience.
"""

from __future__ import annotations

import sys


def add_parser(sub) -> None:
    p = sub.add_parser("features", help="the point-in-time feature layer")
    fs = p.add_subparsers(dest="features_command", required=True)

    b = fs.add_parser("build", help="compute and cache the feature panel")
    b.add_argument("--rebuild", action="store_true", help="ignore any cached panel")
    b.add_argument("--families", default=None,
                   help="comma-separated subset: price,events,fundamental,macro")
    b.set_defaults(func=cmd_build)

    ls = fs.add_parser("list", help="every feature and how often it is populated")
    ls.add_argument("--sort", default="overall", choices=["overall", "feature", "recent"])
    ls.set_defaults(func=cmd_list)

    ck = fs.add_parser(
        "check",
        help="rebuild with the future deleted and assert the past is bit-identical")
    ck.add_argument("--cut-at", default="2016-12-30",
                    help="the date to pretend is today")
    ck.set_defaults(func=cmd_check)

    sh = fs.add_parser("show", help="one feature's distribution through time")
    sh.add_argument("feature")
    sh.add_argument("--annual", action="store_true", help="year ends only")
    sh.set_defaults(func=cmd_show)


def cmd_build(args) -> int:
    from . import build_features
    from .panel import FAMILIES
    families = (tuple(f.strip() for f in args.families.split(",")) if args.families
                else FAMILIES)
    fp = build_features(families=families, rebuild=args.rebuild)
    print(f"  {fp.n_features} features x {len(fp.rows)} rebalance dates "
          f"x {len(fp.security_ids)} securities")
    print(f"  {fp.meta['start']} .. {fp.meta['end']}   "
          f"feature_version={fp.meta['feature_version']}")
    print()
    print("  Nothing here is trustworthy until the leakage test passes:")
    print("    python -m sp500lab features check")
    return 0


def cmd_list(args) -> int:
    import pandas as pd
    from . import build_features
    cov = build_features().coverage()
    cov = cov.sort_values(args.sort, ascending=(args.sort == "feature"))
    out = cov.copy()
    for c in ("overall", "recent"):
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(out.to_string(index=False))
    print()
    print("`overall` counts every (date, security) cell including dates a name was not")
    print("in the index, so ~80% is the practical ceiling. `recent` is the last year,")
    print("which is what a live strategy would actually see.")
    return 0


def cmd_check(args) -> int:
    from .panel import check_leakage, format_leakage
    report = check_leakage(cut_at=args.cut_at)
    print(format_leakage(report))
    if not report["ok"]:
        print()
        print("  A feature that changes when later data is deleted was reading later",
              file=sys.stderr)
        print("  data. Every backtest that used it is void. Fix the feature, do not",
              file=sys.stderr)
        print("  widen the tolerance.", file=sys.stderr)
    return 0 if report["ok"] else 1


def cmd_show(args) -> int:
    import numpy as np
    import pandas as pd
    from . import build_features

    fp = build_features()
    if not fp.has(args.feature):
        print(f"  unknown feature {args.feature!r}", file=sys.stderr)
        print(f"  available: {', '.join(fp.names)}", file=sys.stderr)
        return 1

    mat = fp.matrix(args.feature)
    rows = []
    for i, date in enumerate(fp.dates):
        vals = mat[i][np.isfinite(mat[i])]
        if not len(vals):
            continue
        rows.append({"date": str(date), "n": len(vals),
                     "p5": np.percentile(vals, 5), "median": np.median(vals),
                     "p95": np.percentile(vals, 95)})
    df = pd.DataFrame(rows)
    if args.annual:
        df = df[df["date"].str.slice(5, 7) == "12"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.round(4).to_string(index=False))
    return 0
