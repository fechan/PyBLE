"""
Pyble is a module that helps you work with PHOIBLE distinctive features data from
https://github.com/phoible/dev/tree/master/raw-data/FEATURES

Specifically, this module uses phoible-segments-features.tsv
"""
from typing import List, Dict, TextIO
import pandas as pd
import numpy as np

### Initialize data as a DataFrame
ALL_SEGMENTS = (pd.read_csv("phoible-segments-features.tsv", sep="\t", encoding="utf-8")
    .set_index("segment", drop=False))

class Inventory:
    """
    An inventory of segments and their featural specifications
    """
    def __init__(self, segments_df: pd.DataFrame):
        """
        Initialize an Inventory from a DataFrame of segments and their featural specifications

        :param segments_df: a DataFrame of segments
        """
        self.segments = segments_df

    @classmethod
    def from_ipa(cls, ipa: List[str]) -> "Inventory":
        """
        Initialize an Inventory from a list of IPA strings representing segments

        :param ipa: list of IPA segments
        """
        return cls(ALL_SEGMENTS[ALL_SEGMENTS["segment"].isin(ipa)])

    @classmethod
    def from_all_phoible(cls) -> "Inventory":
        """
        Initialize an inventory containing every unique segment in PHOIBLE
        """
        return cls(ALL_SEGMENTS)

    def dump_csv(self, file: TextIO):
        """
        Dump this inventory's segment data as CSV

        :param file: file to dump data to
        """
        self.segments.to_csv(file, encoding="utf-8", index=False)

    def matching_segments(self, feature_matrix: Dict[str, str]) -> "Inventory":
        """
        Get the natural class of segments in the inventory that match the feature matrix

        :param feature_matrix: Dict that maps distinctive features to their values
        :returns: an Inventory containing segments of the natural class
        """
        return Inventory(self.segments[np.logical_and.reduce([self.segments[k] == v for k,v in feature_matrix.items()])])

    def transform(self, feature_matrix: Dict[str, str]) -> "Inventory":
        return Inventory(self.segments.assign(**feature_matrix))
