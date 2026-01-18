import argparse
import os
import openai
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


# Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
ENC_MODEL_NAME = "text-embedding-3-small"
RES_MODEL_NAME = "gpt-4o"
MAX_TOKENS = 512


def generate_business_desc(company_name):
    prompt = f"Provide one sentence business description for the company named '{company_name}'."
    try:
        response = openai.OpenAI().responses.create(
            model=RES_MODEL_NAME,
            input = prompt,
        )
        return response.output_text
    except Exception as e:
        print(f"Error generating description for {company_name}: {e}")
        return ""


def generate_macro_exposures(company_name):
    prompt = f"Provide one sentence of macro relevant exposure for the company named '{company_name}'." 
    try:
        response = openai.OpenAI().responses.create(
            model=RES_MODEL_NAME,
            input = prompt,
        )
        return response.output_text
    except Exception as e:
        print(f"Error generating default drivers for {company_name}: {e}")
        return ""


def get_embedding(texts, model=ENC_MODEL_NAME):
    response = openai.embeddings.create(input=texts, model=model)
    return [d.embedding for d in response.data]


def main(args):
    company_mapping = pd.read_csv(os.path.join(args.mapping_dir, "company_mapping_BICS2020.csv")).drop_duplicates()
    industry_mapping = pd.read_csv(os.path.join(args.mapping_dir, "industry_mapping_BICS2020.csv"))

    main_df = company_mapping.merge(
        industry_mapping, 
        left_on=["INDUSTRY_SECTOR_NUMBER", "INDUSTRY_GROUP_NUMBER", "INDUSTRY_SUBGROUP_NUMBER"], 
        right_on = ["Industry_sector_num", "Industry_group_num", "Industry_subgroup_num"],
        how="left"
    )

    company_info = []

    for idx, row in tqdm(main_df.iterrows(), total=main_df.shape[0]):
        u3_id = row['U3_ID']
        company_name = row['CompanyName']
        industry_name = row['Industry_sector']
        industry_group = row['Industry_group']
        industry_subgroup = row['Industry_subgroup']
        
        company_desc = generate_business_desc(company_name)
        default_drivers = generate_macro_exposures(company_name)

        company_info.append({
            "U3_ID": u3_id,
            "CompanyName": company_name,
            "Industry_sector": industry_name,
            "Industry_group": industry_group,
            "Industry_subgroup": industry_subgroup,
            "Business_description": company_desc,
            "Default_drivers": default_drivers
        })
    
    company_info_df = pd.DataFrame(company_info)
    company_info_df.to_csv(args.company_info_df, index=False)

    company_info_df = pd.read_csv(args.company_info_df)
    
    for idx, row in tqdm(company_info_df.iterrows(), total=company_info_df.shape[0]):
        u3_id = row['U3_ID']
        company_name = row['CompanyName']
        industry_name = row['Industry_sector']
        industry_group = row['Industry_group']
        industry_subgroup = row['Industry_subgroup']
        company_desc = row['Business_description']
        default_drivers = row['Default_drivers']
        embedding_text = f"{company_name}. {industry_name}. {industry_group}. {industry_subgroup}. {company_desc}. {default_drivers}"
        embbeding = np.array(get_embedding([embedding_text]))
        np.save(args.mapping_dir + f"/embeddings/company_embeddings/{u3_id}.npy", embbeding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--company_info_df",
    )
    parser.add_argument(
        "--mapping_dir",
    )
    args = parser.parse_args()
    main(args)