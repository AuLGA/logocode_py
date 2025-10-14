"""
This module defines the base model class for LoGoMOD.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

import pandas as pd
from tqdm import tqdm
import pickle
from glob import glob

import warnings
warnings.filterwarnings(action='ignore')

from scipy.stats import chisquare
import uuid

import duckdb

import time


class LoGoMOD:
    """
    Base class for LoGoMOD 
    """

    def __init__(self, sample_file_path: str, 
                 family_margins_file_path: str, 
                 individual_margins_file_path: str,
                 output_dir: str,
                 start_year: int = 2025,
                 end_year: int = 2030,
                 **kwargs: Any) -> None:
        """
        Initializes the LoGoMOD model with a sample file path and additional parameters.

        Args:
            sample_file_path (str): Path to the sample file.
            **kwargs: Additional keyword arguments for model configuration.
        """
        self.sample_file_path = sample_file_path
        self.family_margins_file_path = family_margins_file_path
        self.individual_margins_file_path = individual_margins_file_path
        self.output_dir = output_dir
        self.start_year = start_year
        self.output_file = None
        self.end_year = end_year
        self.config = kwargs

    def _drop_families(self, candidate, fmg, permutations, join_vars):
        cols = candidate.columns
        candidate_mg = candidate.groupby(join_vars).size().reset_index(name="Count")
        candidate_mg = pd.concat([candidate_mg, permutations]).groupby(join_vars).agg({"Count": "sum"}).reset_index()

        candidate_mg["diff"] = candidate_mg["Count"] - fmg["Count"] 
        p = candidate_mg[candidate_mg["diff"] > 0]
        k = pd.merge(p, candidate, on=join_vars, how="inner")
        candidate = candidate[~candidate["ABSFID"].isin(k["ABSFID"])]

        candidate = pd.concat([candidate, k.groupby(join_vars).apply(lambda x: x.sample(x["Count"].iat[0] - x["diff"].iat[0]))[cols]])
        
        candidate_mg = candidate.groupby(join_vars).size().reset_index(name="Count")
        candidate_mg = pd.concat([candidate_mg, permutations]).groupby(join_vars).agg({"Count": "sum"}).reset_index()

        return candidate[cols], candidate_mg
    
    def _construct_families(self, lga, families, fmg_lga, persons, permutations, con, join_vars):
        candidate_f = pd.DataFrame(columns=families.columns)
        #print(f"Processing {lga}")
        fmg = fmg_lga[fmg_lga["LGA (EN)"] == lga]
        fmg = fmg[join_vars + ["Count"]]

        total_families = fmg["Count"].sum()

        #permutations_lga = permutations[permutations["LGA (EN)"] == lga]
        #permutations_lga = permutations_lga[join_vars + ["Count"]]

        fmg_i = fmg

        while candidate_f.shape[0]/total_families < 0.99:#total_families != candidate_f.shape[0]:

            candidate = families.sample(n=total_families - candidate_f.shape[0], replace=False)
            candidate, candidate_mg = self._drop_families(candidate, fmg_i, permutations, join_vars)
            candidate_f = pd.concat([candidate_f, candidate])
            fmg_i = fmg_i - candidate_mg
            
            #print(candidate_f.shape[0]/total_families, f"counter: {counter}")
            #candidate_fmg = candidate_f.groupby(join_vars).size().reset_index(name="Count")

            #candidate_fmg = pd.concat([candidate_fmg, permutations]).groupby(join_vars).agg({"Count": "sum"}).reset_index()

            # try:
            #     fmg_test = pd.merge(fmg, candidate_fmg, on=join_vars, how="right")
            #     res = chisquare(f_obs = candidate_fmg["Count"], f_exp = fmg_test["Count"])
            #     print(f"F stat: {res.statistic}, pvalue: {res.pvalue}, Significant at 5% level: {res.pvalue <= 0.01}")
            #     converged = res.pvalue <= 0.005
            #     flag = not converged
            # except:
            #     pass
            #     #print(f"Not there yet : {e}")

        candidate = families.sample(n=total_families - candidate_f.shape[0], replace=False)
        candidate_f = pd.concat([candidate_f, candidate])
                
        candidate_fmg = candidate_f.groupby(join_vars).size().reset_index(name="Count")

        candidate_f["UID"] = candidate_f["ABSFID"].apply(lambda x: uuid.uuid1())
        #candidate_f["UID"] = candidate_f["UID"].astype(str)

        reconstructed_families = pd.merge(candidate_f, persons, on="ABSFID", how="inner")
        lga_name = lga.replace(" (", "_").replace(" ", "_").replace("-", "_").replace(")", "").replace(".", "").replace(",", "").replace("'", "")

        reconstructed_families["LGA"] = lga

        # Move units forward by n years

        stepped_population = reconstructed_families.copy()
        stepped_population["YEAR"] = 2021

        reconstructed_families = pd.DataFrame(columns=stepped_population.columns)

        for year in range(2022, self.end_year + 1):
            stepped_population = self._dynamics_base_lga(year, lga, stepped_population)
            stepped_population["YEAR"] = year

            if year >= self.start_year:
                reconstructed_families = pd.concat([reconstructed_families, stepped_population])

        con.execute(f"CREATE TABLE reconstructed_families_{lga_name} AS SELECT * FROM reconstructed_families")  
        con.execute(f"CREATE TABLE reconstructed_families_mg_{lga_name} AS SELECT * FROM candidate_fmg")      
    
        #store.put(f"reconstructed_families_{lga_name}", reconstructed_families)
        #store.put(f"reconstructed_families_mg_{lga_name}", candidate_fmg)

    def _agep_recode(self, x):
        if x < 5: 
            return 1
        elif x < 10:
            return 2
        elif x < 15:
            return 3
        elif x < 20:
            return 4
        elif x < 25:
            return 5
        else:
            return x - 19

    def generate_synthetic_population(self) -> str:
        """
        Generates a quick synthetic population based on the sample file.

        Returns:
            str: Path to the h5 output file containing the synthetic population.
        """
        # ## Reconciling Family level constraints

        recode_family_margins = {
            "CACF Count of All Children in Family" : {
                'One child in family' : 1,
                'Two children in family' : 2,
                'Three children in family' : 3,
                'Four children in family' : 4,
                'Five children in family' : 5,
                'Six or more children in family' : 5, # Recoded to avoid conflicts with sample (only accounted for 0.16% percent of families)
                'Not applicable' : 7,
            },
            "CPRF Count of Persons in Family" : {
                'Two persons in family' : 2,
                'Three persons in family' : 3,
                'Four persons in family' : 4,
                'Five persons in family' : 5,
                'Six or more persons in family' : 6,
                'Not applicable' : 7,
            },
            "FINASF Total Family Income as Stated (weekly)" : {
                'Negative income' : 1,
                'Nil income' : 2,
                '$1-$149 ($1-$7,799)' : 3,
                '$150-$299 ($7,800-$15,599)' : 4,
                '$300-$399 ($15,600-$20,799)' : 5,
                '$400-$499 ($20,800-$25,999)' : 6, 
                '$500-$649 ($26,000-$33,799)' : 7,
                '$650-$799 ($33,800-$41,599)' : 8, 
                '$800-$999 ($41,600-$51,999)' : 9,
                '$1,000-$1,249 ($52,000-$64,999)' : 10, 
                '$1,250-$1,499 ($65,000-$77,999)' : 11,
                '$1,500-$1,749 ($78,000-$90,999)' : 12, 
                '$1,750-$1,999 ($91,000-$103,999)' : 13,
                '$2,000-$2,499 ($104,000-$129,999)' : 14,
                '$2,500-$2,999 ($130,000-$155,999)' : 15,
                '$3,000-$3,499 ($156,000-$181,999)' : 16, 
                '$3,500-$3,999 ($182,000-$207,999)' : 17,
                '$4,000-$4,499 ($208,000-$233,999)' : 18, 
                '$4,500-$4,999 ($234,000-$259,999)' : 19,
                '$5,000-$5,999 ($260,000-$311,999)' : 20, 
                '$6,000-$7,999 ($312,000-$415,999)' : 21,
                '$8,000 or more ($416,000 or more)' : 22, 
                'All incomes not stated' : 23,
                'Not applicable' : 24,
            },
            "1-digit level FMCF Family Composition" : {
                'Couple family with no children' : 1,
                'Couple family with children' : 2,
                'One parent family' : 3,
                'Other family'  : 4,
                'Not applicable' : 5,
            },
            "SSCF Same-Sex Couple Indicator" : {
                'Male same-sex couple' : 1,
                'Female same-sex couple' : 2,
                'Opposite-sex couple' : 3,
                'Not applicable' : 4,
            },
        }

        rename_family_columns = {
            "CACF Count of All Children in Family" : "CACF",
            "CPRF Count of Persons in Family" : "CPRF",
            "FINASF Total Family Income as Stated (weekly)" : "FINASF",
            "1-digit level FMCF Family Composition" : "FMCF",
            "SSCF Same-Sex Couple Indicator" : "SSCF",
        }

        # family_margins = []

        # for file in tqdm(glob("./Data/Census Margins/Family/*.csv")):
        #     margin = pd.read_csv(file, index_col=False)
        #     margin.rename(columns={"Unnamed: 2": "Count"}, inplace=True)

        #     margin.ffill(inplace=True)

        #     margin = margin[margin["LGA (EN)"] != "Total"]

        #     key = margin.columns[1]

        #     margin[key] = margin[key].map(recode_family_margins[key])

        #     margin.rename(columns={
        #         key: rename_family_columns[key]
        #     }, inplace=True)

        #     family_margins.append(margin)

        fm = pd.read_csv(self.family_margins_file_path, index_col=False)

        fm.rename(columns={"Unnamed: 5": "Count"}, inplace=True)
        fm.ffill(inplace=True)

        for column in fm.columns:
            if column == "LGA (EN)" or column == "Count":
                continue
            fm[column] = fm[column].map(recode_family_margins[column])
            fm.rename(columns={
                column: rename_family_columns[column]
            }, inplace=True)

        fm.columns

        join_vars = ['FMCF', 'SSCF', 'CACF', 'FINASF']
        lga_join_vars = ['LGA (EN)'] + join_vars

        fmg_lga = fm.groupby(lga_join_vars).agg({"Count": "sum"}).reset_index()

        families = pd.read_csv(f"{self.sample_file_path}/CENSUS_2021_BASIC_family.csv", index_col=False)

        fmcf_recode = {
            1 : 1,
            2: 2,
            3: 2,
            4: 2,
            5: 3,
            6: 3,
            7: 3,
            8: 4,
            9: 5,
        }

        families["FMCF_old"] = families["FMCF"].copy()
        families["FMCF"] = families["FMCF"].map(fmcf_recode)

        permutations = fmg_lga[lga_join_vars].drop(columns="LGA (EN)").drop_duplicates()
        permutations["Count"] = 0
        
        # ## Reconciling individual level constraints

        recode_individual_margins = {
                'AGE5P Age in Five Year Groups': {
                '0-4 years' : 1,
                '5-9 years' : 2,
                '10-14 years' : 3,
                '15-19 years' : 4,
                '20-24 years' : 5,
                '25-29 years' : 6,
                '30-34 years' : 7,
                '35-39 years' : 8,
                '40-44 years' : 9,
                '45-49 years' : 10,
                '50-54 years' : 11,
                '55-59 years' : 12,
                '60-64 years' : 13,
                '65-69 years' : 14,
                '70-74 years' : 15,
                '75-79 years' : 16,
                '80-84 years' : 17,
                '85-89 years' : 18, 
                '90-94 years' : 18, 
                '95-99 years' : 18,
            '100 years and over' : 18,
        },
            'GNGP Public/Private Sector': {
                'National Government' : 1,
                'State/Territory Government' : 2,
                'Local Government' : 3,
                'Private sector' : 4,
                'Not stated' : 5,
                'Not applicable' : 6,
                'Overseas visitor' : 7,
        },
            '1-digit level HEAP Level of Highest Educational Attainment': {
                'Postgraduate Degree Level' : 1,
                'Graduate Diploma and Graduate Certificate Level' : 2,
                'Bachelor Degree Level' : 3,
                'Advanced Diploma and Diploma Level' : 4,
                'Certificate III & IV Level' : 5,
                'Secondary Education - Years 10 and above' : 6,
                'Certificate I & II Level' : 7,
                'Secondary Education - Years 9 and below' : 8,
                'Supplementary Codes' : 9,
                'Not stated' : 10,
                'Not applicable' : 11,
                'Overseas visitor' : 12,
        },
            '1-digit level INDP Industry of Employment': {
                'Agriculture, Forestry and Fishing' : 1,
                'Mining' : 2,
                'Manufacturing' : 3,
                'Electricity, Gas, Water and Waste Services' : 4,
                'Construction' : 5,
                'Wholesale Trade' : 6,
                'Retail Trade' : 7,
                'Accommodation and Food Services' : 8,
                'Transport, Postal and Warehousing' : 9,
                'Information Media and Telecommunications' : 10,
                'Financial and Insurance Services' : 11,
                'Rental, Hiring and Real Estate Services' : 12,
                'Professional, Scientific and Technical Services' : 13,
                'Administrative and Support Services' : 14,
                'Public Administration and Safety' : 15,
                'Education and Training' : 16,
                'Health Care and Social Assistance' : 17,
                'Arts and Recreation Services' : 18,
                'Other Services' : 19,
                'Inadequately described' : 20,
                'Not stated' : 21,
                'Not applicable' : 22,
                'Overseas visitor' : 23,
        },
            'INGP Indigenous Status': {
                'Non-Indigenous' : 1,
                'Aboriginal' : 2,
                'Torres Strait Islander' : 2,
                'Both Aboriginal and Torres Strait Islander' : 2,
                'Not stated' : 3,
                'Overseas visitor' : 4,
        },
            'LFSP Labour Force Status': {
                'Employed, worked full-time' : 1,
                'Employed, worked part-time' : 1,
                'Employed, away from work' : 1,
                'Unemployed, looking for full-time work' : 2,
                'Unemployed, looking for part-time work' : 2,
                'Not in the labour force' : 3,
                'Not stated' : 4,
                'Not applicable' : 5,
                'Overseas visitor' : 6,
        },
            '1-digit level OCCP Occupation': {
                'Managers' : 1,
                'Professionals' : 2,
                'Technicians and Trades Workers' : 3,
                'Community and Personal Service Workers' : 4,
                'Clerical and Administrative Workers' : 5,
                'Sales Workers' : 6,
                'Machinery Operators and Drivers' : 7,
                'Labourers' : 8,
                'Inadequately described' : 9,
                'Not stated' : 10,
                'Not applicable' : 11,
                'Overseas visitor' : 12,
        },
            'SEXP Sex': {
                'Male' : 1,
                'Female' : 2,
        },
            'STUP Full-Time/Part-Time Student Status': {
                'Not attending' : 1,
                'Full-time student' : 2,
                'Part-time student' : 3,
                'Institution (TYPP) stated, full-time/part-time status (STUP) not stated' : 4,
                'Both not stated - both institution (TYPP) and full-time/part-time status (STUP) not stated' : 4,
                'Overseas visitor' : 5,
        },
        }

        rename_individual_columns = {
            'AGE5P Age in Five Year Groups': "AGE5P",
            'GNGP Public/Private Sector': "GNGP",
            '1-digit level HEAP Level of Highest Educational Attainment': "HEAP",
            '1-digit level INDP Industry of Employment': "INDP",
            'INGP Indigenous Status': "INGP",
            'LFSP Labour Force Status': "LFSP",
            '1-digit level OCCP Occupation': "OCCP",
            'SEXP Sex': "SEXP",
            'STUP Full-Time/Part-Time Student Status': "STUP",
        }

        # individual_margins = []

        # for file in tqdm(glob("./Data/Census Margins/Individual/*.csv")):
        #     margin = pd.read_csv(file, index_col=False)
        #     margin.rename(columns={"Unnamed: 2": "Count"}, inplace=True)

        #     margin.ffill(inplace=True)

        #     margin = margin[margin["LGA (EN)"] != "Total"]

        #     key = margin.columns[1]

        #     margin[key] = margin[key].map(recode_individual_margins[key])

        #     margin.rename(columns={
        #         key: rename_individual_columns[key]
        #     }, inplace=True)

        #     individual_margins.append(margin)

        persons = pd.read_csv(f"{self.sample_file_path}/CENSUS_2021_BASIC_person.csv", index_col=False)

        persons["AGE5P"] = persons["AGEP"].apply(self._agep_recode)

        im = pd.read_csv(self.individual_margins_file_path, index_col=False)

        im.rename(columns={im.columns[-1]: "Count"}, inplace=True)
        im.ffill(inplace=True)

        for column in im.columns:
            if column == "LGA (EN)" or column == "Count":
                continue
            im[column] = im[column].map(recode_individual_margins[column])
            im.rename(columns={
                column: rename_individual_columns[column]
            }, inplace=True)

        permutations_ind = im.drop(columns="LGA (EN)")
        permutations_ind["Count"] = 0 
        permutations_ind.drop_duplicates(inplace=True)

        persons.columns.sort_values()

        persons = persons[["ABSFID", "ABSPID", "AGE5P", "GNGP", "HRSP", "HSCP", "INCP", "INDP", "INGP", "LFSP", "MTWP", "OCCP", "OCSKP", "SEXP", "STUP"]]

        # ## Save to HDF store

        timestr = time.strftime("%Y_%m_%d-%H%M%S")

        # store = pd.HDFStore(f"{self.output_dir}/logomod_{timestr}.h5")

        # store.put("families", families)
        # store.put("persons", persons)

        # ## Save to DuckDB 

        con = duckdb.connect(database=f"{self.output_dir}/logomod_{timestr}.duckdb", read_only=False)

        con.execute("CREATE TABLE families AS SELECT * FROM families")
        con.execute("CREATE TABLE persons AS SELECT * FROM persons")

        # ## Slide Sampling
                

        lgas = fmg_lga["LGA (EN)"].unique()
        #lgas = [lga.replace(" ", "_").replace("-", "_").replace("(", "_").replace(")", "").replace(".", "").replace(",", "").replace("'", "") for lga in lgas]

        lgas = [lga for lga in lgas if "Migratory" not in lga]
        lgas = [lga for lga in lgas if "Unincorp." not in lga]

        for lga in tqdm(lgas):
            self._construct_families(lga, families, fmg_lga, persons, permutations, con, join_vars)

        lgas = pd.DataFrame({
            "LGA": lgas
        })

        lgas["LGA Name"] = lgas["LGA"].apply(lambda x: x.replace(" (", "_").replace(" ", "_").replace("-", "_").replace(")", "").replace(".", "").replace(",", "").replace("'", ""))
        
        # store.put("lgas", lgas)

        #store.close()

        con.execute("CREATE TABLE lgas AS SELECT * FROM lgas")

        return f"{self.output_dir}/logomod_{timestr}.duckdb"

    def _dynamics_base_lga(self, target_year: int, lga: str, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies base dynamics to the synthetic population.

        Args:
            target_year (int): The target year for applying dynamics.
            input_db (Optional[str]): Path to the input duckdb file. If None, uses the latest generated file.

        Returns:
            pd.DataFrame: The updated synthetic population DataFrame.
        """

        
        for col in ['FINASF', 'INCP']:
            input_df[col] = input_df[col] * np.random.normal(1.03, 0.005, input_df.shape[0])

        # Update population by growing by 2% annually

        # select 2% families randomly
        families = input_df[["UID", "ABSFID"]].drop_duplicates()
        n_families = int(families.shape[0] * 0.02)
        families_to_grow = families.sample(n=n_families, replace=False)

        new_families = input_df[input_df["UID"].isin(families_to_grow["UID"])].copy()

        new_families["UID"] = new_families["ABSFID"].apply(lambda x: uuid.uuid1())

        input_df["AGE5P"] = input_df['AGE5P'].apply(lambda x: x + 1 if (x < 18) and (np.random.rand() < 0.2) else 18)

        input_df = pd.concat([input_df, new_families])

        return input_df





        

        
            

