"""FEFO pick-planning of net weights across parsed GS1 scans.

Each scan is fed through :func:`app.gs1.parse` and must then contain:

* exactly one AI 01 (GTIN),
* exactly one AI 17 (expiry date; day ``00`` denotes the last day of the
  month, per the GS1 General Specifications),
* zero or one AI 10 (batch/lot) -- optional,
* exactly one weight element of the 310n (kg) / 320n (lb) family.

Any parser failure, missing/duplicated required AI or multiple weight
elements make the whole request fail with
:class:`app.summary.WeightSummaryError` whose ``detail`` names the
offending scan index -- the same 422 shape as ``/weight-summary``.

Valid scans are aggregated by (GTIN, batch, AI 17 value) in first-seen
order; every weight is rebuilt with :class:`decimal.Decimal` from the
original six digits and converted with the fixed factors of
:mod:`app.summary`.  Each demand is then satisfied FEFO: the GTIN's
groups are walked by ascending *actual* expiry date (day ``00`` -> last
day of the month), ties keep the batches' first-appearance order (the
sort is stable), and the last group reached may be allocated partially.

Every demand may carry a non-negative ``min_remaining_days`` (default
0): given the request's ``as_of`` date, only groups whose actual expiry
date is on or after ``as_of + min_remaining_days`` may serve that
demand -- the boundary date itself is eligible.  Excluded groups never
contribute to ``fulfilled`` (the shortfall stays in ``shortage``); their
combined weight per GTIN is reported as ``excluded_weight``.  A negative
``min_remaining_days`` fails the whole request with
:class:`DemandValidationError` naming the offending ``demand_index``.

A demand may additionally carry ``package_mode``:

* ``"splittable"`` (default) -- the aggregated-group behavior described
  above, with the last group split partially; ``overage`` is always
  zero.
* ``"whole"`` -- every eligible *scan* is an indivisible package, so
  scans are never aggregated and no package may be split.  If some
  subset of the eligible packages reaches ``quantity``, the subset with
  the smallest overage is chosen; ties go to the lexicographically
  smallest tuple of indices into the FEFO ordering (actual expiry date,
  then original scan order).  If no subset reaches ``quantity`` every
  eligible package is taken and the remainder is the shortage.  The
  chosen packages return one allocation each, in FEFO order.  A demand
  with more than :data:`WHOLE_MODE_MAX_PACKAGES` eligible packages
  fails with :class:`DemandValidationError` naming ``demand_index``;
  zero-weight scans never count as packages (they contribute nothing,
  mirroring the splittable skip of zero-weight groups).

All conversion, aggregation, comparison and allocation arithmetic stays
in :class:`Decimal`; rounding to six places happens only when the
response is rendered.
"""

from __future__ import annotations

import calendar
from bisect import bisect_left
from datetime import date, timedelta
from decimal import Decimal, localcontext

from .gs1 import GS1ParseError, Element, decode_weight, parse
from .summary import (
    ARITHMETIC_PRECISION,
    ScanValidationError,
    WeightSummaryError,
    _failure,
    _to_target,
)

REQUIRED_AI_TEXT = (
    "exactly one AI 01 (GTIN), exactly one AI 17 (expiry date), "
    "at most one AI 10 (batch/lot), and exactly one weight AI 310n/320n"
)

# Whole-package selection is an exact subset-sum search; bound the input
# so the meet-in-the-middle work stays trivially fast (2**15 sums each
# side) and the response stays actionable.
WHOLE_MODE_MAX_PACKAGES = 30


class DemandValidationError(Exception):
    """A demand violated the request shape (e.g. a duplicated GTIN).

    ``detail`` is the complete 422 payload: ``demand_index`` plus the
    validation error under ``error``, mirroring the scan-level shape.
    """

    def __init__(self, detail: dict):
        super().__init__(detail["error"]["message"])
        self.detail = detail


def _expiry_date(yymmdd: str) -> date:
    """Actual calendar date of an AI 17 value; day ``00`` = last day of month."""
    year = 2000 + int(yymmdd[0:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    if day == 0:
        day = calendar.monthrange(year, month)[1]
    return date(year, month, day)


def _threshold(as_of: date, min_remaining_days: int) -> date:
    """Earliest eligible actual expiry date for one demand."""
    try:
        return as_of + timedelta(days=min_remaining_days)
    except OverflowError:
        # A threshold beyond any representable date excludes everything.
        return date.max


def _extract(
    index: int, elements: list[Element]
) -> tuple[str, str | None, str, Decimal, str]:
    """Pull (GTIN, batch or None, AI 17 value, exact weight, unit) from one scan."""
    gtins = [e for e in elements if e.ai == "01"]
    expiries = [e for e in elements if e.ai == "17"]
    batches = [e for e in elements if e.ai == "10"]
    weights = [e for e in elements if e.unit is not None]

    if len(gtins) != 1:
        if not gtins:
            raise ScanValidationError(
                "MISSING_AI",
                f"scan {index}: missing required AI 01 (GTIN)",
                expected=REQUIRED_AI_TEXT,
            )
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: AI 01 (GTIN) appears {len(gtins)} times; exactly one is required",
            expected="exactly one AI 01 (GTIN)",
            found=f"{len(gtins)} AI 01 elements",
            offset=gtins[1].offset,
        )

    if len(expiries) != 1:
        if not expiries:
            raise ScanValidationError(
                "MISSING_AI",
                f"scan {index}: missing required AI 17 (expiry date)",
                expected=REQUIRED_AI_TEXT,
            )
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: AI 17 (expiry date) appears {len(expiries)} times; exactly one is required",
            expected="exactly one AI 17 (expiry date)",
            found=f"{len(expiries)} AI 17 elements",
            offset=expiries[1].offset,
        )

    if len(batches) > 1:
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: AI 10 (batch/lot) appears {len(batches)} times; at most one is allowed",
            expected="at most one AI 10 (batch/lot)",
            found=f"{len(batches)} AI 10 elements",
            offset=batches[1].offset,
        )

    if len(weights) != 1:
        if not weights:
            raise ScanValidationError(
                "MISSING_AI",
                f"scan {index}: missing required weight AI 310n/320n",
                expected=REQUIRED_AI_TEXT,
            )
        ais = ", ".join(e.ai for e in weights)
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: {len(weights)} weight elements ({ais}); "
            "exactly one 310n/320n is required",
            expected="exactly one weight AI 310n (kg) or 320n (lb)",
            found=ais,
            offset=weights[1].offset,
        )

    weight = weights[0]
    value = decode_weight(weight.value, int(weight.ai[3]))
    return (
        gtins[0].value,
        batches[0].value if batches else None,
        expiries[0].value,
        value,
        weight.unit,
    )


def _check_demands(demands: list[tuple[str, Decimal, int, str]]) -> None:
    """Reject negative thresholds and GTINs requested by more than one demand."""
    seen: dict[str, int] = {}
    for demand_index, (gtin, _quantity, min_remaining_days, _mode) in enumerate(demands):
        if min_remaining_days < 0:
            raise DemandValidationError(
                {
                    "demand_index": demand_index,
                    "error": {
                        "error": "INVALID_MIN_REMAINING_DAYS",
                        "message": (
                            f"demand {demand_index}: min_remaining_days must be "
                            f"a non-negative integer, got {min_remaining_days}"
                        ),
                        "expected": "non-negative integer min_remaining_days",
                        "found": str(min_remaining_days),
                    },
                }
            )
        if gtin in seen:
            first = seen[gtin]
            raise DemandValidationError(
                {
                    "demand_index": demand_index,
                    "error": {
                        "error": "DUPLICATE_GTIN",
                        "message": (
                            f"demand {demand_index}: GTIN '{gtin}' is already "
                            f"requested by demand {first}; each GTIN may "
                            "appear only once"
                        ),
                        "expected": "unique GTIN across all demands",
                        "found": gtin,
                    },
                }
            )
        seen[gtin] = demand_index


def _subset_sums(weights: list[Decimal]) -> list[Decimal]:
    """All 2**len(weights) subset sums, grouped by mask: index ``mask``
    holds the sum of the weights whose bit is set in ``mask``.
    """
    sums = [Decimal(0)]
    for weight in weights:
        # Snapshot length: extending while iterating sums would loop forever.
        size = len(sums)
        sums.extend(sums[i] + weight for i in range(size))
    return sums


def _can_make(
    target: Decimal,
    left_sums: list[Decimal],
    right_sums: list[Decimal],
) -> bool:
    """Whether a left/right subset-sum pair (as produced by the split
    :func:`_subset_sums` calls) adds up to exactly ``target``.
    """
    if target < 0:
        return False
    if target == 0:
        return True
    right = set(right_sums)
    for left in left_sums:
        if left > target:
            continue
        if target - left in right:
            return True
    return False


def _choose_whole_packages(
    packages: list[dict], quantity: Decimal
) -> tuple[list[int], Decimal]:
    """Pick whole packages for one demand.

    Returns ``(indices, fulfilled)`` where ``indices`` index into
    ``packages`` (already ordered FEFO).  If some subset reaches
    ``quantity``, the subset with the smallest overage wins; ties resolve
    to the lexicographically smallest index tuple.  Otherwise every
    package is taken.

    The optimum is found with meet-in-the-middle subset sums (each half
    has at most 15 packages) and the lexicographic tie-break is rebuilt
    greedily: an index is included exactly when a valid completion with
    that prefix and the fixed overage still exists -- the earliest
    possible indices therefore win, which is precisely index-tuple
    lexicographic order.
    """
    weights = [package["weight"] for package in packages]
    n = len(weights)
    total = sum(weights, Decimal(0))

    if total < quantity:
        return list(range(n)), total

    split = (n + 1) // 2
    left, right = weights[:split], weights[split:]
    left_sums = _subset_sums(left)
    right_sums = _subset_sums(right)
    right_values = sorted(right_sums)

    # Smallest reachable total >= quantity: for every left sum take the
    # smallest right sum that closes the gap (bisect over exact Decimals).
    best_total = total
    for left_sum in left_sums:
        if left_sum > best_total:
            continue
        pos = bisect_left(right_values, quantity - left_sum)
        if pos < len(right_values):
            candidate = left_sum + right_values[pos]
            if candidate < best_total:
                best_total = candidate
                if best_total == quantity:
                    break

    # Rebuild the lexicographically smallest subset (in index-tuple
    # order) that sums to best_total.  Fixed left/right sum pairs let
    # each feasibility probe stay a small meet-in-the-middle check.
    chosen: list[int] = []
    fixed_sum = Decimal(0)
    for i, weight in enumerate(weights):
        remaining = weights[i + 1 :]
        trial_sum = fixed_sum + weight
        split_trial = (len(remaining) + 1) // 2
        trial_left_sums = _subset_sums(remaining[:split_trial])
        trial_right_sums = _subset_sums(remaining[split_trial:])
        if _can_make(best_total - trial_sum, trial_left_sums, trial_right_sums):
            chosen.append(i)
            fixed_sum = trial_sum
    return chosen, best_total


def pick_plan(
    scans: list[str],
    target_unit: str,
    demands: list[tuple[str, Decimal, int, str]],
    as_of: date,
) -> dict:
    """Aggregate scan weights per GTIN and satisfy each demand FEFO.

    ``demands`` is a list of ``(gtin, quantity, min_remaining_days,
    package_mode)`` tuples with ``quantity`` a :class:`Decimal` already
    expressed in ``target_unit``, ``min_remaining_days`` a non-negative
    integer and ``package_mode`` ``"splittable"`` (aggregated groups,
    last group may be split) or ``"whole"`` (each eligible scan is an
    indivisible package); a GTIN may appear only once.  Only scans whose
    actual expiry date is on or after ``as_of + min_remaining_days`` may
    serve a demand.
    Returns ``{"target_unit": ..., "demands": [{gtin, quantity,
    fulfilled, shortage, overage, excluded_weight, allocations}, ...]}``
    in demand order, where every numeric value is an unrounded
    :class:`Decimal` in the target unit, ``excluded_weight`` is the
    total weight of the demand's GTIN held in scans/groups the threshold
    ruled out, and each allocation is ``{"batch", "expiry", "weight"}``
    with ``expiry`` the raw AI 17 value of the source.  In splittable
    mode allocations are per group; in whole mode each allocation is one
    selected scan and the list is ordered FEFO.
    """
    _check_demands(demands)

    groups: dict[tuple[str, str | None, str], dict] = {}
    order: list[tuple[str, str | None, str]] = []
    # Whole mode consumes individual scans rather than aggregated groups.
    packages_by_gtin: dict[str, list[dict]] = {}

    with localcontext() as ctx:
        ctx.prec = ARITHMETIC_PRECISION
        for index, scan in enumerate(scans):
            try:
                elements = parse(scan)
            except GS1ParseError as exc:
                raise _failure(index, exc) from exc
            try:
                gtin, batch, expiry, weight, unit = _extract(index, elements)
            except ScanValidationError as exc:
                raise _failure(index, exc) from exc

            converted = _to_target(weight, unit, target_unit)
            key = (gtin, batch, expiry)
            if key not in groups:
                order.append(key)
                groups[key] = {
                    "gtin": gtin,
                    "batch": batch,
                    "expiry": expiry,
                    "expiry_date": _expiry_date(expiry),
                    "available": Decimal(0),
                }
            groups[key]["available"] += converted

            if converted > 0:
                # A zero-weight scan cannot move any demand; excluding it
                # mirrors the splittable skip of zero-weight groups and
                # keeps it out of the package cap.
                packages_by_gtin.setdefault(gtin, []).append(
                    {
                        "batch": batch,
                        "expiry": expiry,
                        "expiry_date": _expiry_date(expiry),
                        "weight": converted,
                        "scan_index": index,
                    }
                )

        by_gtin: dict[str, list[dict]] = {}
        for key in order:
            by_gtin.setdefault(key[0], []).append(groups[key])
        for gtin_groups in by_gtin.values():
            # FEFO: ascending actual expiry date.  The sort is stable, so
            # groups expiring on the same date keep their first-seen order.
            gtin_groups.sort(key=lambda group: group["expiry_date"])
        for gtin_packages in packages_by_gtin.values():
            # Packages sort by actual expiry, ties by original scan order.
            gtin_packages.sort(key=lambda package: (package["expiry_date"], package["scan_index"]))

        results = []
        for demand_index, (gtin, quantity, min_remaining_days, mode) in enumerate(demands):
            threshold = _threshold(as_of, min_remaining_days)

            if mode == "whole":
                results.append(
                    _plan_whole(
                        demand_index,
                        gtin,
                        quantity,
                        threshold,
                        packages_by_gtin.get(gtin, []),
                    )
                )
                continue

            remaining = quantity
            allocations = []
            excluded = Decimal(0)
            # Groups are sorted by ascending actual expiry, so every
            # group the threshold rules out (expiry before as_of +
            # min_remaining_days) forms a prefix of the walk; their
            # weight is summed into excluded_weight and they never
            # contribute to fulfilled.
            for group in by_gtin.get(gtin, []):
                if group["expiry_date"] < threshold:
                    excluded += group["available"]
                    continue
                if remaining <= 0:
                    break
                take = min(remaining, group["available"])
                if take <= 0:
                    continue  # group of zero-weight scans contributes nothing
                allocations.append(
                    {
                        "batch": group["batch"],
                        "expiry": group["expiry"],
                        "weight": take,
                    }
                )
                remaining -= take
            results.append(
                {
                    "gtin": gtin,
                    "quantity": quantity,
                    "fulfilled": quantity - remaining,
                    "shortage": remaining,
                    "overage": Decimal(0),
                    "excluded_weight": excluded,
                    "allocations": allocations,
                }
            )

    return {"target_unit": target_unit, "demands": results}


def _plan_whole(
    demand_index: int,
    gtin: str,
    quantity: Decimal,
    threshold: date,
    packages: list[dict],
) -> dict:
    """Satisfy one demand with indivisible packages (``package_mode="whole"``)."""
    eligible = [p for p in packages if p["expiry_date"] >= threshold]
    if len(eligible) > WHOLE_MODE_MAX_PACKAGES:
        raise DemandValidationError(
            {
                "demand_index": demand_index,
                "error": {
                    "error": "TOO_MANY_PACKAGES",
                    "message": (
                        f"demand {demand_index}: package_mode 'whole' allows at "
                        f"most {WHOLE_MODE_MAX_PACKAGES} eligible packages, "
                        f"found {len(eligible)}"
                    ),
                    "expected": (
                        f"at most {WHOLE_MODE_MAX_PACKAGES} eligible packages "
                        "for package_mode 'whole'"
                    ),
                    "found": str(len(eligible)),
                },
            }
        )

    excluded_weight = sum(
        (p["weight"] for p in packages if p["expiry_date"] < threshold),
        Decimal(0),
    )

    chosen, fulfilled = _choose_whole_packages(eligible, quantity)
    allocations = [
        {
            "batch": eligible[i]["batch"],
            "expiry": eligible[i]["expiry"],
            "weight": eligible[i]["weight"],
        }
        for i in chosen
    ]
    shortage = quantity - fulfilled
    if shortage < 0:
        shortage = Decimal(0)
    return {
        "gtin": gtin,
        "quantity": quantity,
        "fulfilled": fulfilled,
        "shortage": shortage,
        "overage": fulfilled - quantity if fulfilled > quantity else Decimal(0),
        "excluded_weight": excluded_weight,
        "allocations": allocations,
    }
