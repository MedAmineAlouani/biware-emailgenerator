import pandas as pd
import streamlit as st
from chains import Chain
from lead_qual import feature_engineering
import joblib


def process_raw_data(raw_data):
    try:
        og_data = pd.read_csv('app/resource/Lead Qualification Data.csv')
        og_data.drop(columns=['Output'],inplace=True)
        matched_indices = og_data[og_data['Contact'].isin(raw_data['Contact'])].index.tolist()
        og_data_eng = feature_engineering(og_data.copy())
        matched_data_eng = og_data_eng.loc[matched_indices]
        st.info("Predicting qualified leads using the model...")
        model = joblib.load("lead_qual_model.pkl")
        predictions = model.predict(matched_data_eng)
        matched_data_eng['Predictions'] = predictions
        st.write("Prediction value counts:")
        st.write(matched_data_eng['Predictions'].value_counts())

        qualified_leads = matched_data_eng[matched_data_eng['Predictions'] == 1]
        st.write(f"Qualified leads found: {len(qualified_leads)}")
        if qualified_leads.empty:
            st.warning("No qualified leads were found.")
            return pd.DataFrame()  # Return an empty DataFrame

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
        st.error(f"An Error Occurred: {e}")
        return pd.DataFrame()

def create_streamlit_app(llm):
    # Add a custom header image/logo
    st.image("app/resource/Biware.png", use_column_width=True)  # Replace with your image path or URL
    st.title("📧 Biware Lead Qualification & Email Generator")

    # Step 1: Upload raw CSV file
    file_upload = st.file_uploader("Upload your raw CSV file with client data:", type="csv")

    # Step 2: Select email type
    email_type = st.selectbox(
        "Select the type of email to generate:",
        ["Cold Outreach", "Follow-Up", "Thank You Email", "Proposal Email"]
    )

    if file_upload:
        try:
            # Load raw CSV data
            raw_data = pd.read_csv(file_upload)

            # Display the head of the uploaded raw data
            st.info("Uploaded Raw Data Preview:")
            st.write(raw_data.head())

            # Ensure required columns are present in raw data
            required_columns = ["Contact", "Fonction", "Société", "Secteur", "Pays"]
            if not all(col in raw_data.columns for col in required_columns):
                st.error(f"The CSV file must include the following columns: {', '.join(required_columns)}")
                return

            # Step 3: Process raw data through Lead Qualification
            st.info("Processing data with Lead Qualification...")
            email_data = process_raw_data(raw_data)

            # Display qualified leads for email generation
            if email_data.empty:
                st.warning("No qualified leads were found in the uploaded data.")
                return

            st.success("Lead Qualification completed. Displaying qualified leads:")
            st.write(email_data.head(20))
            # Step 4: Generate Emails
            st.info("Generating emails...")
            all_emails = []
            for _, client in email_data.iterrows():
                job_description = {
                    "Contact": client["Name"],
                    "Job": client["Job"],
                    "Company": client["Company"],
                    "EmailType": email_type  # Pass email type to the LLM
                }
                email = llm.write_mail(job_description)
                all_emails.append(email)

                # Display each generated email
                st.subheader(f"Generated {email_type} for {client['Name']}:")
                st.code(email, language="markdown")

            # Allow download of all emails as .txt
            if all_emails:
                emails_combined = "\n\n".join(all_emails)
                st.download_button(
                    label="📥 Download All Emails as .txt",
                    data=emails_combined,
                    file_name=f"{email_type.lower().replace(' ', '_')}_emails.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email Generator for Biware", page_icon="📧")
    create_streamlit_app(chain)
