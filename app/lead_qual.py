import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import pickle


# Feature engineering


def extract_seniority(fonction):
    fonction = str(fonction).lower()
    senior_keywords = [
        "directeur", "directrice", "chef", "responsable", "manager", "head",
        "lead", "c-level", "ceo", "cfo", "cio", "cto", "president", "coo",
        "dga", "dg", "administrateur", "chief", "directeur général"
    ]
    mid_keywords = [
        "spécialiste", "specialist", "ingénieur", "consultant", "analyst",
        "supervisor", "coordinateur", "coordinator", "project manager",
        "chargé", "chargée", "développeur", "developer", "senior analyst"
    ]
    junior_keywords = [
        "stagiaire", "junior", "trainee", "intern", "assistant", "associate"
    ]
    if any(keyword in fonction for keyword in senior_keywords):
        return "senior"
    elif any(keyword in fonction for keyword in mid_keywords):
        return "mid"
    elif any(keyword in fonction for keyword in junior_keywords):
        return "junior"
    else:
        return "unknown"


def apply_extract_seniority(data):
    data['Seniority'] = data["Fonction"].apply(extract_seniority)
    return data


def simplify_sector(data):
    sector_mapping = {
        'Telco': 'Telecom',
        'Télécommunications': 'Telecom',
        'Internet Provider': 'Telecom',
        'Banque': 'Finance',
        'Microfinance': 'Finance',
        'Microfinance ': 'Finance',
        'Financial Services': 'Finance',
        'financial services': 'Finance',
        'Capital Markets/Securities': 'Finance',
        'Assurance': 'Finance',
        'Energie': 'Energy',
        'Oil & Energy': 'Energy',
        'Oil Field Services Company': 'Energy',
        'Civilian Government': 'Government/Public',
        'Public Administration': 'Government/Public',
        'Government Administration': 'Government/Public',
        'Grande distribution': 'Retail',
        'Retail ': 'Retail',
        'Boissons': 'Retail',
        'Restaurants': 'Retail',
        'Cosmetique': 'Retail',
        'Hôtel': 'Hospitality',
        'Gambling': 'Hospitality',
        'Compagnie aerienne': 'Transport',
        'Transport de marchandises maritime': 'Transport',
        'Transport Services': 'Transport',
        'IT': 'Technology/IT',
        'Consultancy': 'Technology/IT',
        'Services': 'Technology/IT',
        'Industrie': 'Manufacturing/Industry',
        'Machinerie': 'Manufacturing/Industry',
        'Construction': 'Manufacturing/Industry',
        'Laboratoire pharmaceutique': 'Manufacturing/Industry',
        'automobile': 'Manufacturing/Industry',
        'Agroalimentaire': 'Agriculture',
        'Real Estate': 'Real Estate',
        'Entreprise': 'Other',
        'Other - Unsegmented': 'Other',
        'unknown': 'Other'
    }
    data["Simplified_Sector"] = data["Secteur"].map(sector_mapping).fillna('Other')
    return data


def simplify_region(data):
    region_mapping = {
        'Algerie': 'North Africa',
        'Libye': 'North Africa',
        'Maroc': 'North Africa',
        'Morocco': 'North Africa',
        'Tunisie': 'North Africa',
        'Mauritanie': 'North Africa',
        'Benin': 'West Africa',
        'Togo': 'West Africa',
        'Togo ': 'West Africa',
        'Burkina Faso': 'West Africa',
        'Guinée': 'West Africa',
        'Guinée Equatoriale': 'West Africa',
        'Mali': 'West Africa',
        'Mali ': 'West Africa',
        "Côte d'Ivoire": 'West Africa',
        'Senegal': 'West Africa',
        'Sénégal': 'West Africa',
        'Sénégal ': 'West Africa',
        'Cameroun': 'Central Africa',
        'Cameroun ': 'Central Africa',
        'Cameroun / RDC': 'Central Africa',
        'Congo Brazza': 'Central Africa',
        'République Démocratique du Congo': 'Central Africa',
        'République Du Congo': 'Central Africa',
        'République du Congo': 'Central Africa',
        'Gabon': 'Central Africa',
        'République Centrafricaine ': 'Central Africa',
        'Ethiopie': 'East Africa',
        'Burundi': 'East Africa',
        'Ile Maurice': 'Indian Ocean Islands',
        'Mauritius': 'Indian Ocean Islands',
        'Réunion': 'Indian Ocean Islands',
        'Réunion ': 'Indian Ocean Islands',
        'Nigeria': 'Sub-Saharan Africa',
        'Tchad': 'Sub-Saharan Africa',
        'RDC Kinshasa': 'Sub-Saharan Africa',
        'France ': 'Europe',
        'Espagne': 'Europe'
    }

    data["Region"] = data["Pays"].map(region_mapping).fillna('Unknown')
    return data


def add_company_frequency(data):
    company_frequency = data["Société"].value_counts()
    data["Company_Frequency"] = data["Société"].map(company_frequency)
    return data


def simplify_company_name(name):
    if pd.isnull(name):
        return name
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s]', '', name)
    common_words = ['group', 'inc', 'company', 'corporation', 'ltd', 'limited', 'sarl', 'spa', 'sa']
    for word in common_words:
        name = re.sub(rf'\b{word}\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def apply_simplify_company_name(data):
    data['Société'] = data['Société'].apply(simplify_company_name)
    return data


def simplify_job_title(job_title):
    if pd.isnull(job_title):
        return job_title
    job_title = job_title.lower()  # Convert to lowercase
    job_title = re.sub(r'[^a-z0-9\s]', '', job_title)  # Remove non-alphanumeric characters
    job_title = re.sub(r'\s+', ' ', job_title).strip()  # Remove extra spaces
    return job_title


def apply_simplify_job_title(data):
    data['Fonction'] = data['Fonction'].apply(simplify_job_title)
    return data


def categorize_job_title(title):
    title = str(title).lower()  # Normalize case
    # IT/Tech roles
    if any(keyword in title for keyword in ['it', 'tech', 'digital', 'cio', 'cto', 'information', 'systems']):
        return 'IT/Tech'
    # Finance roles
    elif any(keyword in title for keyword in ['finance', 'account', 'cfo', 'credit', 'treasury', 'risk']):
        return 'Finance'
    # Marketing/Sales roles
    elif any(keyword in title for keyword in ['marketing', 'sales', 'commercial', 'crm']):
        return 'Marketing/Sales'
    # Human Resources roles
    elif any(keyword in title for keyword in ['hr', 'human', 'recruitment', 'talent', 'resources']):
        return 'Human Resources'
    # Operations/Administration roles
    elif any(keyword in title for keyword in ['operations', 'logistics', 'admin', 'supply', 'process']):
        return 'Operations/Administration'
    # Leadership/Executive roles
    elif any(keyword in title for keyword in ['director', 'chief', 'executive', 'ceo', 'head']):
        return 'Leadership/Executive'
    # Default category
    return 'Other'


def apply_categorize_job_title(data):
    data['Job_Category'] = data['Fonction'].apply(categorize_job_title)
    return data


def is_large_company(name):
    if pd.isnull(name):
        return 0
    name = name.lower()
    large_companies = [
        "orange", "renault", "total", "vodafone", "societe generale", "citibank", "bnp paribas",
        "standard chartered", "toyota", "bgfi bank", "attijariwafa bank", "microsoft", "google",
        "amazon", "airbus", "carrefour", "shell", "deutsche bank", "bp", "ecobank", "uba"
    ]
    large_patterns = ["group", "holding", "corporation", "international", "inc", "corporate"]
    if any(large in name for large in large_companies):
        return 1
    elif any(pattern in name for pattern in large_patterns):
        return 1
    else:
        return 0


def apply_is_large_company(data):
    data['is_large_company'] = data['Société'].apply(is_large_company)
    return data


# Load original label encoders
with open('label_encoders.pkl', 'rb') as f:
    original_encoders = pickle.load(f)


def map_with_original_encoding(data, column_name, encoder):
    if column_name not in data:
        raise ValueError(f"Column {column_name} not found in the data.")

    # Get the current maximum value in the encoder
    max_value = len(encoder.classes_)

    # Create a mapping function for the column
    def map_value(x):
        if x in encoder.classes_:
            return encoder.transform([x])[0]  # Use the encoded value for known categories
        else:
            return max_value  # Assign a new value for unseen categories

    # Apply the mapping function to the column
    data[column_name] = data[column_name].apply(map_value)

    return data




def feature_engineering(data):
    data = apply_extract_seniority(data)
    data = simplify_sector(data)
    data = simplify_region(data)
    data = add_company_frequency(data)
    data = apply_simplify_company_name(data)
    data = apply_simplify_job_title(data)
    data = apply_categorize_job_title(data)
    data = apply_is_large_company(data)

    # Map with original encoders
    data = map_with_original_encoding(data, 'Seniority', original_encoders['Seniority'])
    data = map_with_original_encoding(data, 'Region', original_encoders['Region'])
    data = map_with_original_encoding(data, 'Job_Category', original_encoders['Job_Category'])
    data = map_with_original_encoding(data, 'Simplified_Sector', original_encoders['Simplified_Sector'])
    data = map_with_original_encoding(data, 'Société', original_encoders['Société'])
    data = map_with_original_encoding(data, 'Pays', original_encoders['Pays'])

    data.drop(columns=['Secteur', 'Contact', 'Fonction'], inplace=True)

    return data

