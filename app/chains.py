import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def scrape_website(self, url):
        """
        Scrape the given website URL and return the cleaned text.
        """
        try:
            loader = WebBaseLoader([url])
            page_content = loader.load().pop().page_content
            return page_content.strip()
        except Exception as e:
            raise Exception(f"Failed to scrape the website: {e}")

    def extract_jobs(self, cleaned_text):
        """
        Extract job postings from the scraped text in JSON format.
        """
        prompt_extract = PromptTemplate.from_template(
            f"""
            ### SCRAPED TEXT FROM WEBSITE:
            {cleaned_text}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: 
            `role`, `experience`, `skills`, `description`, `contact_name`, `contact_job_title`, and `company`.
            If `contact_name` is missing, generate it as "Responsible Hiring Manager".
            If `contact_job_title` is missing, infer it from the context of the job posting.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]


    def display_job_description(self, job):
        """
        Display the job details in a clear, formatted style.
        """
        print("\n================= Job Details =================")
        print(f"Role: {job.get('role', 'Not Provided')}")
        print(f"Company: {job.get('company', 'Not Provided')}")
        print(f"Contact Person: {job.get('contact_name', 'Responsible Hiring Manager')} ({job.get('contact_job_title', 'Not Provided')})")
        print(f"Experience Required: {job.get('experience', 'Not Provided')}")
        print(f"Key Skills: {', '.join(job.get('skills', [])) if job.get('skills') else 'Not Provided'}")
        print("\nJob Description:")
        print(job.get('description', 'Not Provided'))
        print("================================================\n")

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

    def write_mail_url(self, job, links=None,email_type="Email de Prospection"):

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

        links = links or []
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
            Make sure you utilize the description of the job provided to emphasis on how Biware can help with this job.
            Your job is to {instruction}.   
            The email must be in French. 
            Do not create fictional email addresses at the end.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": ", ".join(links)})
        return res.content

if __name__ == "__main__":
    chain = Chain()
    url = input("Enter the job posting URL: ")
    try:
        print("Scraping the website...")
        scraped_text = chain.scrape_website(url)
        jobs = chain.extract_jobs(scraped_text)
        for idx, job in enumerate(jobs):
            print(f"\nJob #{idx + 1}:\n")
            chain.display_job_description(job)
            generate_email = input("Generate an email for this job? (yes/no): ").strip().lower()
            if generate_email == "yes":
                email = chain.write_mail(job)
                print("\nGenerated Email:")
                print(email)
                print("\n================================================\n")
    except Exception as e:
        print(f"An error occurred: {e}")
