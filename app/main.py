import pandas as pd
import streamlit as st
from chains import Chain
from lead_qual import feature_engineering
import joblib


def process_raw_data(raw_data):
    try:
        og_data = pd.read_csv('app/resource/Lead Qualification Data.csv')
        og_data.drop(columns=['Output'], inplace=True)
        raw_data.index = range(og_data.index.max() + 1, og_data.index.max() + 1 + len(raw_data))
        og_data = pd.concat([og_data, raw_data])
        og_data_eng = feature_engineering(og_data.copy())
        st.info("Prédiction des prospects qualifiés à l'aide du modèle AI...🧠 ")
        model = joblib.load("lead_qual_model.pkl")
        predictions = model.predict(og_data_eng)
        og_data_eng['Predictions'] = predictions
        raw_data_predictions = og_data_eng.loc[raw_data.index, 'Predictions']
        st.write("Prediction value counts (raw_data only):")
        st.write(raw_data_predictions.value_counts())
        qualified_leads = og_data.loc[raw_data_predictions.index][raw_data_predictions == 1]
        st.write(f"Prospects qualifiés trouvés : {len(qualified_leads)}")
        if qualified_leads.empty:
            st.warning("Aucun prospect qualifié n'a été trouvé. 😔")
            return pd.DataFrame()
        email_data = og_data.loc[qualified_leads.index, ['Contact', 'Fonction', 'Société']]
        email_data.rename(columns={
            'Contact': 'Name',
            'Fonction': 'Job',
            'Société': 'Company'
        }, inplace=True)
        # Save email data for debugging or further processing
        email_data.to_csv('app/resource/email_data.csv', index=False)
        return email_data
    except Exception as e:
        st.error(f"Une erreur est survenue :{e}")
        return pd.DataFrame()


def create_streamlit_app(llm):
    # Add a custom header image/logo
    st.image("app/resource/Biware.png", use_column_width=True)
    st.title("📧 Biware Lead Qualification & Email Generator")

    # Step 1: Choose mode of operation
    mode = st.radio(
        "Choisissez un mode d'opération :",
        options=["Qualification des prospects et génération d'emails", "Passer la qualification des prospects","Scraper un site web pour générer des emails", "Saisir les détails manuellement"]
    )

    # Step 2: Select email type
    email_type = st.selectbox(
        "Sélectionnez le type d'email à générer :",
        ["Email de Prospection", "Email de Relance", "Remerciement", "Proposition", "Proposition de formation"]
    )



    if mode == "Qualification des prospects et génération d'emails":
        # Upload raw CSV file
        file_upload = st.file_uploader("Téléchargez votre fichier CSV brut avec les données des clients :", type="csv")

        if file_upload:
            try:
                # Loading the raw CSV data
                raw_data = pd.read_csv(file_upload)

                # Displaying the uploaded raw data
                st.info("Aperçu des données brutes téléchargées :")
                st.write(raw_data.head())

                # Make sure data is in the right format
                required_columns = ["Contact", "Fonction", "Société", "Secteur", "Pays"]
                if not all(col in raw_data.columns for col in required_columns):
                    st.error(f"Le fichier CSV doit inclure les colonnes suivantes : {', '.join(required_columns)}")
                    return

                # Process raw data through Lead Qualification
                st.info("Traitement des données avec la qualification des prospects...⚙️")
                email_data = process_raw_data(raw_data)

                # Display qualified leads for email generation
                if email_data.empty:
                    st.warning("Aucun prospect qualifié n'a été trouvé dans les données téléchargées. 😞 ")
                    return

                st.success("Qualification des prospects terminée. Affichage des prospects qualifiés :")
                st.write(email_data.head(20))

                # Generate Emails
                st.info("Génération des emails... ")
                generate_emails(email_type,llm,mode,email_data)
            except Exception as e:
                st.error(f"An Error Occurred: {e}")



    elif mode == "Passer la qualification des prospects":
        # Upload raw CSV file
        file_upload = st.file_uploader("Upload your raw CSV file with client data:", type="csv")

        if file_upload:
            try:
                # Loading the raw CSV data
                raw_data = pd.read_csv(file_upload)

                # Ensure required columns are present
                required_columns = ["Contact", "Fonction", "Société"]
                if not all(col in raw_data.columns for col in required_columns):
                    st.error(f"Le fichier CSV doit inclure les colonnes suivantes : {', '.join(required_columns)}")
                    return

                # Use raw_data directly for email generation
                st.info("Passage de la qualification des prospects. Utilisation des données téléchargées pour la génération des emails.")
                email_data = raw_data.rename(columns={
                    'Contact': 'Name',
                    'Fonction': 'Job',
                    'Société': 'Company'
                })
                generate_emails(email_type,llm,mode,email_data)
            except Exception as e:
                st.error(f"An Error Occurred: {e}")



    elif mode == "Saisir les détails manuellement":
        # Manual input for Name, Job, and Company
        st.info("Saisissez les détails pour la génération manuelle des emails :")
        name = st.text_input("Nom")
        job = st.text_input("Fonction")
        company = st.text_input("Société")

        if st.button("Générer un email 📩 "):
            if name and job and company:
                email_data = pd.DataFrame([{'Name': name, 'Job': job, 'Company': company}])
                generate_emails(email_type,llm,mode,email_data)
            else:
                st.error("Veuillez remplir tous les champs : Nom, Poste et Entreprise.")



    elif mode == "Scraper un site web pour générer des emails":
        url = st.text_input("Entrez l'URL de la page de carrière ou d'emploi :")
        if st.button("Scraper et générer des emails"):
            try:
                st.info("Scraping en cours... 🌐")
                scraped_text = llm.scrape_website(url)
                jobs = llm.extract_jobs(scraped_text)

                if not jobs:
                    st.warning("Aucun job n'a été extrait de cette URL. Vérifiez le contenu.")
                    return

                for index, job in enumerate(jobs):
                    # Display job details and generate email
                    st.subheader(f"Détails du job :")
                    st.write(f"**Rôle**: {job.get('role', 'Non fourni')}")
                    st.write(f"**Entreprise**: {job.get('company', 'Non fourni')}")
                    st.write(
                        f"**Contact**: {job.get('contact_name', 'Responsable du recrutement')} ({job.get('contact_job_title', 'Non fourni')})")
                    st.write(f"**Expérience requise**: {job.get('experience', 'Non fourni')}")
                    st.write(
                        f"**Compétences**: {', '.join(job.get('skills', [])) if job.get('skills') else 'Non fourni'}")
                    st.write("**Description**:")
                    st.write(job.get('description', 'Non fourni'))

                    # Generate email automatically after displaying job details
                    st. info(f"Génération de l'email pour le job {job.get('role', 'Non fourni')}... 📩")
                    email= llm.write_mail_url(job, email_type)
                    st.code(email, language="markdown")
            except Exception as e:
                st.error(f"Une erreur est survenue lors du scraping ou de la génération de l'email : {e}")

def generate_emails(email_type, llm, mode, email_data=pd.DataFrame([]), job=[]):
    try:
        all_emails = []
        for _, client in email_data.iterrows():
            job_description = {
                "Contact": client["Name"],
                "Job": client["Job"],
                "Company": client["Company"],
                "EmailType": email_type
            }
            email = llm.write_mail(job_description)
            all_emails.append(email)

            # Display each generated email
            st.subheader(f"Generated {email_type} for {client['Name']}:")
            st.code(email, language="markdown")

        # Download of all emails as .txt
        if all_emails:
            emails_combined = "\n\n".join(all_emails)
            st.download_button(
                label="📥 Télécharger tous les emails au format .txt",
                data=emails_combined,
                file_name=f"{email_type.lower().replace(' ', '_')}_emails.txt",
                mime="text/plain"
            )
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la génération de l'email : {e}")


if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Email Generator for Biware", page_icon="📧")
    create_streamlit_app(chain)
