import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-90b-vision-preview")

    def write_mail(self, job):
        email_type = job.get("EmailType", "Cold Outreach")

        if email_type == "Email de Prospection":
            instruction = "Write a professional cold email introducing Biware."
        elif email_type == "Email de Relance":
            instruction = "Write a follow-up email referencing a previous communication."
        elif email_type == "Remerciement":
            instruction = "Write a thank you email for the client's time."
        elif email_type == "Proposition":
            instruction = "Write an email presenting Biware's detailed proposal."
        elif email_type == "Proposition de formation":
            instruction = "Write an email where biware proposes a training service"

        else:
            instruction = "Write a general professional email."

        prompt_email = PromptTemplate.from_template(
            f"""
            ### JOB DESCRIPTION:
            {{job_description}}

            ### INSTRUCTION:
            You are Biware Worker Example, a business development executive at Biware.
            Biware is a young and dynamic consulting and system integration company specialized in Data Management & Modern Analytics. 
            Biware is composed of technical & business experts mainly IT & Statistic (Data Scientist) engineers.
            Biware is based in Tunisia with representatives in Lagos and Paris.
            Biware delivers Customer Intelligence, Risk Management, Fraud and Compliance, Demand Forecasting, and Model Analytics Solutions 
            to Large Businesses for Finance, Telecommunication, Retail & Energy/Utilities companies using the most advanced analytics software.
            Your job is to {instruction}.

            The email has to be in French.
            Do not create fictional emails adresses to insert at the end.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job)})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
