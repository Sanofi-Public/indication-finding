"""a class to read UMLS data stored on s3"""

import logging
import os

import boto3
import numpy as np
import pandas as pd

from src.utils.aws_utils import split_s3_file

logger = logging.getLogger()
project_path = os.path.split(os.path.split(__file__)[0])[0]
LOCAL_DATA_LOCATION = os.path.join(project_path, ".UMLS_TMP")


class UmlsRRFReader:
    """
    A class that handles the loading of data from UMLS into Pandas Dataframes.
    See this documentation about the files that this class reads:
    https://www.ncbi.nlm.nih.gov/books/NBK9685/
    """

    def __init__(
        self,
        remote_data_location,
        local_data_location=LOCAL_DATA_LOCATION,
    ):
        """
        Keyword Arguments :
            remote_data_location: A string containing the location of the UMLS RRF data files on s3
            local_data_location: A folder on the local machine where data will be downloaded from s3 to.
        """
        self.remote_data_location = remote_data_location
        self.s3_client = boto3.client(
            "s3",
        )

        self.local_data_location = local_data_location
        if not os.path.exists(self.local_data_location):
            os.makedirs(self.local_data_location)

        # The MRFILES contains the column names for each file, along with other metadata.
        self.metadata = pd.read_csv(
            self.download_datafile("MRFILES.RRF"), delimiter="|", header=None
        )
        # The MRCOLS file contains information about columns in each file.
        self.column_descriptions = self.get_frame("MRCOLS.RRF", col_descriptions=False)

    def download_datafile(self, fname):
        """
        Given a filename like "MRFILES.txt" in the UMLS language system,
        download it from the s3 location of the data (self.remote_data_location)
        to the local machine's directory at self.local_data_location.
        """
        local_path = os.path.join(self.local_data_location, fname)

        if self.remote_data_location is None:
            logger.debug(
                "The remote location was none. Assuming files are already at the local destination"
            )
            return local_path

        full_s3_path = os.path.join(self.remote_data_location, fname)
        bucket, key = split_s3_file(full_s3_path)
        if not os.path.exists(local_path):
            logger.info(f"File {fname} was not cached. Downloading to {local_path}.")
            self.s3_client.download_file(bucket, key, local_path)
        else:
            logger.info(f"File {fname} was already cached at {local_path}")
        return local_path

    def get_column_descriptions(self, column_names, filename):
        # query the dataframe of column names for column descriptions
        queries = []
        for column_name in column_names:
            queries.append(f"(COL == '{column_name}' )")
        query_string = f"(FIL == '{filename}') and (" + " or ".join(queries) + ")"
        column_description_frame = self.column_descriptions.query(query_string)
        return column_description_frame

    def get_frame(
        self,
        filename,
        col_descriptions=False,
        usecols=None,
        chunksize=None,
        selection=None,
    ):
        """
        Given a filename of a file in the UMLS, ensure that the file is downloaded to self.local_data_location.
        Load the datafile file as a pandas dataframe.

        Args:
            filename: filename of datafile from UMLS

        Keyword Args:
            col_descriptions: if True, return an additional dataframe containing
                              a description of all columns in the main frame
            usecols: If None, return a dataframe containing all columns in the
                     data file. If it is a list of strings, return a dataframe
                     containing only these columns.
            chunksize: If None: load the entire dataframe at once. 
                       If set to some integer N, load the dataset in
                       chunks of size N, apply the selection, and then
                       concatenate the final frame. This can reduce the overall 
                       memory footprint of loading the dataset, since so little 
                       of it could pass selection.
            selection: a string that will be passed to used to query the dataframe.

        Returns:
            Pandas dataframe
            if col_descriptions is True, then return a tuple of Pandas dataframe and another frame providing descriptions of the columns.

        Example output:
        >>> reader.get_frame("MRREL.RRF", col_descriptions=True, usecols=["CUI2", "CUI1", "SAB", "RELA", "REL"])
        (              CUI1 REL      CUI2            RELA     SAB
            0         C0000005  RB  C0036775             NaN  MSHFRE
            1         C0000005  RB  C0036775             NaN     MSH
            2         C0000039  SY  C0000039  translation_of  MSHSWE
            3         C0000039  SY  C0000039  translation_of  MSHCZE
            4         C0000039  SY  C0000039  translation_of  MSHPOR
            ...            ...  ..       ...             ...     ...
            43842945  C5779496  RB  C4047263     has_version     SRC
            43842946  C5779497  RB  C4047262     has_version     SRC
            43842947  C5779498  RB  C4047261     has_version     SRC
            43842948  C5779499  RB  C5698419     has_version     SRC
            43842949  C5779500  RB  C4047260     has_version     SRC
            [43842950 rows x 5 columns],
                    COL                                                DES  REF  MIN  \
            6        AUI1                   Unique identifier for first atom  NaN    0
            8        AUI2                  Unique identifier for second atom  NaN    0
            26       CUI1                Unique identifier for first concept  NaN    8
            29       CUI2               Unique identifier for second concept  NaN    8
            73        CVF                                  Content view flag  NaN    0
            83        DIR                Source asserted directionality flag  NaN    0
            206      RELA                      Additional relationship label  NaN    0
            213       REL                                 Relationship label  NaN    2
            215        RG                                 Relationship group  NaN    0
            218       RUI                 Unique identifier for relationship  NaN    9
            224       SAB                                Source abbreviation  NaN    2
            237        SL                      Source of relationship labels  NaN    2
            242      SRUI          Source attributed relationship identifier  NaN    0
            247    STYPE1  The name of the column in MRCONSO.RRF that con...  NaN    3
            248    STYPE2  The name of the column in MRCONSO.RRF that con...  NaN    3
            286  SUPPRESS                                  Suppressible flag  NaN    1
        """
        column_names = (
            self.metadata.query(f"@self.metadata[0] == '{filename}'")
            .values[0][2]
            .split(",")
        )

        # if using a subset of the total columns, make sure that they are available.
        if usecols is not None:
            for c in usecols:
                assert c in column_names
        else:
            columns = column_names

        logger.info(f"Reading UMLS file {filename}")
        local_fname = self.download_datafile(filename)
        logger.info(f"Reading local copy of file {filename} at {local_fname}")

        if chunksize is not None:
            logger.info("Loading dataset in chunks.")
            iterable_reader = pd.read_csv(
                local_fname,
                names=column_names,
                delimiter="|",
                index_col=False,
                usecols=usecols,
                chunksize=chunksize,
            )
        else:
            logger.info("Loading dataset all at once")
            iterable_reader = [
                pd.read_csv(
                    local_fname,
                    names=column_names,
                    delimiter="|",
                    index_col=False,
                    usecols=usecols,
                )
            ]

        full_frame = []
        counter = 0
        for frame in iterable_reader:
            if usecols is not None:
                frame = frame[usecols]  # guarantee that the columns are sorted
            if counter % 10 == 0:
                logger.info(f"Read chunk {counter} of file {filename}")
                logger.info(f"Applied selection {selection}")
            if selection is not None:
                frame = frame.query(selection)
            full_frame.append(frame)
            counter += 1

        if len(full_frame) > 1:
            full_frame = pd.concat(full_frame)
        else:
            full_frame = full_frame[0]

        if col_descriptions:
            column_description_frame = self.get_column_descriptions(
                column_names, filename
            )
            return full_frame, column_description_frame
        else:
            return full_frame


def clean_code(code):
    if isinstance(code, str):
        code = code["ICD10CM_CODE"]
    return code.lower().replace(".", "")


def match(code1, code2):
    code1 = clean_code(code1)
    code2 = clean_code(code2)
    return code1 in code2


def is_in_selected_diseases(row, disease_selection):
    icd10cm_code = row["ICD10CM_CODE"]
    for disease in disease_selection:
        if match(disease, icd10cm_code):
            return True
    return False


def merge_drug_forms(form_groups):
    """
    Given a list of sets of drug codes, merge sets of codes together that intersect.

    N.B. This function only does one iteration of merging. It is a helper function for merge_all_drug_form_groups.
    The function merge_all_drug_form_groups should be used in general.

    argugments:
        form_groups: a list of sets of drug codes.
    """
    new_form_groups = []
    already_merged = set()
    for i1, group in enumerate(form_groups):
        if i1 in already_merged:
            continue
        found_intersection = False
        for i2, group2 in enumerate(form_groups[i1 + 1 :]):
            if len(group.intersection(group2)) > 0:
                found_intersection = True
                new_form_groups.append(group.union(group2))
                already_merged.add(i1 + i2 + 1)
                break
        if not found_intersection:
            new_form_groups.append(group)
    return new_form_groups


def merge_all_drug_form_groups(same_form_groups):
    """
    Given a list of sets of drug codes, merge sets of codes together that intersect.
    Repeat the merging until all groups of drugs do not intersect.

    argugments:
        same_form_groups: a list of sets of drug codes.

    Example output:
    >>> list_of_groups = [
        set(["A", "B"]),
        set(["B", "C"]),
        set(["E", "F"]),
        set(["F", "G"]),
        set(["I", "J"]),
        set(["A", "C"]),
    ]
    >>> merge_all_drug_form_groups(list_of_groups)
    [set(["A", "B", "C"]), set(["E", "F", "G"]), set(["I", "J"])]
    """
    merged_form_groups = same_form_groups[:]
    while len(merge_drug_forms(merged_form_groups)) != len(merged_form_groups):
        merged_form_groups = merge_drug_forms(merged_form_groups)
    return merged_form_groups


def generate_may_treat_relationships(
    concepts_frame, relationships_frame, convert_indications_to_vocab
):
    """
    Given a frame of concepts from UMLS, and their relationships, create a dataframe containing drugs and their indications from MedRT. Furthermore, convert the indications to the coding scheme convert_indications_to_vocab.
    Since we mostly work with ICD10CM indications, pass this argument to create our standard drug-indication dataset.
    """
    msh_concepts = concepts_frame.query("SAB == 'MSH'").drop_duplicates(["CUI"])
    rxnorm_concepts = concepts_frame.query("SAB == 'RXNORM'").drop_duplicates(["CUI"])
    may_treat_relationships_in_medrt = relationships_frame.query(
        "SAB == 'MED-RT' and RELA == 'may_treat'"
    )
    logger.info(
        f"{len(may_treat_relationships_in_medrt)} may-treat relationships found in med-rt"
    )

    # join in the mesh description data about the mesh indications that drugs treat
    may_treat_relationships_in_medrt = may_treat_relationships_in_medrt.join(
        msh_concepts.set_index("CUI"), on="CUI1", lsuffix="_relation", rsuffix="_msh"
    )
    may_treat_relationships_in_medrt = may_treat_relationships_in_medrt[
        ["CUI1", "CUI2", "RELA", "STR", "SCUI"]
    ]
    remap = {}
    remap["CUI1"] = "Indication_CUI"
    remap["CUI2"] = "Compound_CUI"
    remap["STR"] = "Indication_MeSH_Label"
    may_treat_relationships_in_medrt = may_treat_relationships_in_medrt.rename(
        columns=remap
    )

    # join in the rxnorm data about the drugs that treat indications
    may_treat_relationships_in_medrt = may_treat_relationships_in_medrt.join(
        rxnorm_concepts.set_index("CUI"),
        on="Compound_CUI",
        lsuffix="_relation",
        rsuffix="_rxnorm",
    )

    # select the subset of columns that we need to work with to build the dataset
    may_treat_relationships_in_medrt = may_treat_relationships_in_medrt[
        [
            "Indication_CUI",
            "Compound_CUI",
            "Indication_MeSH_Label",
            "STR",
            "SCUI_relation",
            "SCUI_rxnorm",
        ]
    ]
    may_treat_relationships_in_medrt = may_treat_relationships_in_medrt.rename(
        columns={
            "STR": "Compound_RXNORM_Label",
            "SCUI_relation": "Indication_MeSH_SCUI",
            "SCUI_rxnorm": "Compound_RXNORM_SCUI",
        }
    )

    # Add to the table the corresponding indications in the converted vocab
    logger.info(f"Converting indications from MeSH to {convert_indications_to_vocab}")
    conversion_concepts_frame = (
        concepts_frame.query(f"SAB == '{convert_indications_to_vocab}'")
        .drop_duplicates(["CUI"])
        .rename(
            columns={
                c: f"{convert_indications_to_vocab}_{c}" for c in concepts_frame.columns
            }
        )
    )
    may_treat_relationships_in_medrt = may_treat_relationships_in_medrt.join(
        conversion_concepts_frame.set_index(f"{convert_indications_to_vocab}_CUI"),
        on="Indication_CUI",
    )

    return may_treat_relationships_in_medrt


def get_prefix(start):
    for i in range(len(start)):
        if start[i].isdigit() and start[i] != "0":
            return min(len(start) - 1, i)
    return len(start) - 1


def expand_icd_code_range(df):
    """
    Expand ICD code range in format c{N}-c{N+K} to individual codes

    args:
        df - dataframe with ICD code ranges
    """
    rows = []
    for index, row in df.iterrows():
        code = row["CLEAN_ICD10CM_CODE"]
        if "-" in code:
            start, end = code.split("-")
            prefixend = get_prefix(start)
            prefix = start[:prefixend]
            for i in range(int(start[prefixend:]), int(end[prefixend:]) + 1):
                new_code = f"{prefix}{i}"
                logger.info(f"Expanding ICD code range {code} to {new_code}")
                new_row = row.copy()
                new_row["CLEAN_ICD10CM_CODE"] = new_code
                rows.append(new_row)
        else:
            rows.append(row)

    return pd.DataFrame(rows)


def create_medrt_dataset(
    concepts_frame,
    relationships_frame,
    disease_selection,
    reference_diseases_vocab="ICD10CM",
):
    """
    Given a frame of concepts from UMLS, their relationships, a list
    of reference diseases to use to select drugs return a dataframe of
    drugs that are known to treat at least one of the diseases
    in the selected_diseases selection.

    args:
        concepts frame: a dataframe of concepts from UMLS. This can be
        generated by calling  UmlsRRFReader().get_frame("MRCONSO.RRF", col_descriptions=True).
        relationships_frame: a dataframe of relations from UMLS. This can be generated by calling 
        UmlsRRFReader().get_frame("", col_descriptions=True). It should have columns 
        like: ["CUI1", "CUI2", "RELA", "SAB"]. disease_selection: a list of strings representing the 
        codes of diseases to use to select drugs. Drugs are selected to treat at least one disease 
        (or sub disease) wihtin this list reference_diseases_vocab: The vobabulary of diseases to use to 
        create the dataset. e.g. 'ICD10CM', 'ICD9CM', etc.

    Example:
    >>> reader = UmlsRRFReader(test=True)
    >>> concepts, concept_descriptions = reader.get_frame("MRCONSO.RRF", col_descriptions=True)
    >>> relationship_frame, relationship_column_descriptions = \
        reader.get_frame("MRREL.RRF", col_descriptions=True, usecols=["CUI1", "CUI2", "RELA", "SAB"])
    >>> disease_selection = ["M05"] # the code for rheumatoid arthuritis.
    >>> create_medrt_dataset(concepts_frame, relationships_frame, disease_selection)
    """

    # select concepts that are only in english.
    concepts_frame = concepts_frame.query("LAT=='ENG'")

    # get all medrt may-treat relationships and select the columns that we need to produce the final dataset
    medrt = generate_may_treat_relationships(
        concepts_frame, relationships_frame, reference_diseases_vocab
    )
    medrt_trimmed = medrt[
        [
            "Compound_RXNORM_Label",
            "Compound_RXNORM_SCUI",
            f"{reference_diseases_vocab}_CODE",
            f"{reference_diseases_vocab}_STR",
            "Compound_CUI",
        ]
    ]

    # get rid of cases where the icd10cm indication could not be found. i.e. the column values are nan after joining.
    isna = medrt_trimmed[f"{reference_diseases_vocab}_STR"].isna()
    logger.info(
        f"{sum(isna)} of {len(medrt_trimmed)} indication-drug pairs are excluded because the MeSH indication couldn't be converted to an {reference_diseases_vocab} indication"
    )
    medrt_trimmed = medrt_trimmed[np.logical_not(isna)]

    medrt_trimmed["selected_diseases_indication"] = medrt_trimmed.apply(
        lambda x: is_in_selected_diseases(x, disease_selection), axis=1
    )
    medrt_trimmed[f"CLEAN_{reference_diseases_vocab}_CODE"] = medrt_trimmed.apply(
        clean_code, axis=1
    )
    selected_diseases_medrt = medrt_trimmed[
        medrt_trimmed["selected_diseases_indication"] == True
    ]
    logger.info(
        f"Of {len(medrt_trimmed)} drug-indication pairs, {len(selected_diseases_medrt)} were found to treat a disease in the disease selection."
    )

    # loop through the relationships and delete the same drugs that have different forms
    logger.info("Dropping drug pairs that share the same form relationship.")
    same_form_relationships = relationships_frame.query(
        "RELA == 'has_form' and SAB == 'RXNORM'"
    )
    same_form_drug_pairs = []
    for i, row in same_form_relationships.iterrows():
        cui1 = row["CUI1"]
        cui2 = row["CUI2"]
        same_form_drug_pairs.append(set([cui1, cui2]))

    # a list of groups of drugs that share the same form.
    merged_form_groups = merge_all_drug_form_groups(same_form_drug_pairs)

    # create a list of unique drugs, avoiding adding duplicate drugs that have the same form group.
    unique_drugs = set()
    for i, row in same_form_relationships.iterrows():
        cui1 = row["CUI1"]
        cui2 = row["CUI2"]
        both_in_group = False
        offending_group = None
        for group in merged_form_groups:
            if cui1 in group and cui2 in group:
                both_in_group = True
                offending_group = group
                break
        if both_in_group:
            member_of_group_already_present = False
            for cui in offending_group:
                if cui in unique_drugs:
                    member_of_group_already_present = True
                    break
            if not member_of_group_already_present:
                unique_drugs.add(cui1)
        else:
            unique_drugs.add(cui1)

    initial_drug_count = len(selected_diseases_medrt["Compound_CUI"].value_counts())
    selected_diseases_medrt = selected_diseases_medrt[
        selected_diseases_medrt["Compound_CUI"].isin(unique_drugs)
    ]
    final_drug_count = len(selected_diseases_medrt["Compound_CUI"].value_counts())
    logger.info(
        f"When accounting for duplicate drug forms, the number of drugs decreased from {initial_drug_count} to {final_drug_count}."
    )

    # expand the dataset to include all indications, regardless of being selected_diseases
    selected_diseases_expanded_medrt = medrt_trimmed[
        medrt_trimmed["Compound_RXNORM_SCUI"].isin(
            set(selected_diseases_medrt["Compound_RXNORM_SCUI"].values)
        )
    ]
    logger.info(
        f"After including all indications of selected drugs, there are {len(selected_diseases_expanded_medrt)} total drug-indication pairs"
    )

    selected_diseases_expanded_medrt = expand_icd_code_range(
        selected_diseases_expanded_medrt
    )
    same_form_relationships = same_form_relationships[["CUI1", "CUI2"]]
    selected_diseases_expanded_medrt = selected_diseases_expanded_medrt.merge(
        same_form_relationships, left_on="Compound_CUI", right_on="CUI1"
    )
    selected_diseases_expanded_medrt = selected_diseases_expanded_medrt.drop(
        columns=["Compound_CUI", "CUI1"]
    )
    selected_diseases_expanded_medrt = selected_diseases_expanded_medrt.rename(
        columns={"CUI2": "Compound_CUI"}
    )
    # return the dataset of drugs and indications
    return selected_diseases_expanded_medrt.reset_index()
