#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021, Juan B Cabral and QuatroPe.
# License: BSD-3-Clause
#   Full Text: https://github.com/quatrope/pyonono/blob/master/LICENSE

# This file is a prototype for the upcoming pyonono benchmark framework

# =============================================================================
# IMPORTS
# =============================================================================

import json
import os
import sys

import attr

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

# =============================================================================
# CLASS
# =============================================================================


@attr.s(frozen=True)
class BMFile:
    _data = attr.ib(repr=False)

    def _to_hashable(self, value):
        converters = {list: tuple, set: frozenset}
        vtype = type(value)
        converter = converters.get(vtype)
        cvalue = converter(value) if converter else value
        return cvalue

    @property
    def machine_info(self):
        return dict(self._data["machine_info"])

    @property
    def params(self):
        params = dict(self._data["benchmarks"][0]["params"])
        params.pop("iteration", None)
        return tuple(params.keys())

    @property
    def iterations(self):
        if "iteration" not in self._data["benchmarks"][0]["params"]:
            return 1
        its = set(p["params"]["iteration"] for p in self._data["benchmarks"])
        return max(its) + 1

    @property
    def size(self):
        return len(self._data["benchmarks"])

    def as_dataframe(self):
        rows = []
        for bm in self._data["benchmarks"]:
            stats = dict(bm["stats"])
            stats.pop("data", None)

            params = {
                f"p.{p}": self._to_hashable(bm["params"][p])
                for p in self.params
            }
            stats.update(params)
            rows.append(stats)

        # to df
        df = pd.DataFrame(rows)

        # reorder cols
        params_cols = [f"p.{p}" for p in self.params]
        data_cols = [c for c in df.columns if c not in params_cols]

        odf = df[params_cols + data_cols]
        return odf

    def boxplot(self, param, ax=None, **box_kwargs):
        df = self.as_dataframe()

        pcolumn = f"p.{param}"
        ax = sns.boxplot(x=pcolumn, y="min", data=df, ax=ax, **box_kwargs)

        ax.set_title(
            f"Benchmark Size {self.size}"
        )
        ax.set_ylabel("Time (secs)")
        ax.set_xlabel(param)

        return ax


# =============================================================================
# FUNCTION
# =============================================================================


def load_file(path):
    with open(path) as fp:
        data = json.load(fp)
    return BMFile(data)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    bm_file = load_file(sys.argv[1])
    for param in bm_file.params:
        new_name = os.path.join(sys.argv[2], f"{param}_bp.pdf")

        fig, ax = plt.subplots()
        bm_file.boxplot(param, ax=ax)

        fig.tight_layout()
        fig.savefig(new_name)

        plt.close()
