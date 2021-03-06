"""
Pyble is a module that helps you work with PHOIBLE distinctive features data from
https://github.com/phoible/dev/blob/master/data/phoible.csv

See also: PHOIBLE repo at https://github.com/phoible/dev
"""
from typing import List, Dict, TextIO
import pandas as pd
import numpy as np

### Initialize data as a DataFrame
# SEGMENT_COL = "segment" # whatever the segment (non-feature) column is named in the data
# ALL_SEGMENTS = (pd.read_csv("phoible-segments-features.tsv", sep="\t", encoding="utf-8")
#     .set_index(SEGMENT_COL, drop=False))

SEGMENT_COL = "Phoneme"
PHOIBLE_DATA = pd.read_csv("phoible.csv", encoding="utf-8")
NONFEATURE_COLS = [col for col in PHOIBLE_DATA.columns if col[0].isupper()]
ALL_SEGMENTS = (PHOIBLE_DATA.drop([col for col in NONFEATURE_COLS if col != SEGMENT_COL], axis=1)
    .drop_duplicates(SEGMENT_COL)
    .set_index(SEGMENT_COL, drop=False))

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
        return cls(ALL_SEGMENTS[ALL_SEGMENTS[SEGMENT_COL].isin(ipa)])

    @classmethod
    def from_all_phoible(cls) -> "Inventory":
        """
        Initialize an inventory containing every unique segment in PHOIBLE
        """
        return cls(ALL_SEGMENTS)

    @classmethod
    def from_phoible_id(cls, phoible_id: int) -> "Inventory":
        """
        Initialize an inventory from its PHOIBLE inventory ID

        :param phoible_id: PHOIBLE inventory ID
        """
        lang_segments = (PHOIBLE_DATA[PHOIBLE_DATA["InventoryID"] == phoible_id]
            .drop([col for col in NONFEATURE_COLS if col != SEGMENT_COL], axis=1)
            .set_index(SEGMENT_COL, drop=False))
        return cls(lang_segments)

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

    def transform(self, feature_matrix: Dict[str, str],
            transform_targets: "Inventory"=None) -> Dict[str, List[str]]:
        """
        Apply the feature values in the feature matrix to the segments in this inventory,
        and get the list of segments among the transform targets that match the new featural
        specifications

        :param feature_matrix: feature matrix to apply to segments in the inventory
        :param transform_targets: (optional) Inventory containing segments that the current
            inventory can be transformed into. Defaults to the current inventory.
        :returns: dict that maps the inventory's segments to a list of matching segments in IPA

        :example: If the inventory consisted of ["p", "b"], then applying the feature matrix
            {"periodicGlottalSource": "+"} to add voicing to each segment would return
            {"p": ["b"], "b": ["b"]}. In other words, p became voiced, and b was already voiced
            so it remained the same.
        """
        if transform_targets is None:
            transform_targets = self

        modified = self.segments.assign(**feature_matrix)
        transform_map = {}
        for segment_index, segment_series in modified.iterrows():
            features = segment_series.drop(SEGMENT_COL)
            transform_map[segment_index] = (transform_targets.matching_segments(features)
                .segments
                .index
                .to_list())
        return transform_map

    def add(self, segment: str, feature_matrix: Dict[str, str], default_value: str="0") -> "Inventory":
        """
        Add a segment to the inventory.
        If the feature matrix of the added segment has features not in the inventory, the features
        are added to the inventory. Existing segments will have an NA value for these features.

        :param segment: IPA of the segment to add
        :param feature_matrix: featural specifications of the added segment
        :param default_value: (optional) default feature value for features in the inventory not
            in the feature matrix
        :returns: Inventory with the given segment added
        """
        feature_matrix[SEGMENT_COL] = segment
        for column in self.segments.columns:
            if column not in feature_matrix:
                feature_matrix[column] = default_value
        added = (self.segments
            .append(feature_matrix, ignore_index=True)
            .set_index(SEGMENT_COL, drop=False))
        return Inventory(added)

    def drop_features(self, features: List[str]) -> "Inventory":
        """
        Drop features from the inventory that are in the given list

        :param features: list of names of features
        :returns: inventory with features dropped
        """
        return Inventory(self.segments.drop(features, axis=1))

    def drop_redundant_features(self) -> "Inventory":
        """
        Drop features where all the segments in the inventory have the same value

        :returns: inventory with redundant features dropped
        """
        output_segments = self.segments.copy()
        for feature in self.segments.columns:
            column_values = self.segments[feature].to_numpy()
            if (column_values[0] == column_values).all():
                output_segments.drop(feature, axis=1, inplace=True)
        return Inventory(output_segments)