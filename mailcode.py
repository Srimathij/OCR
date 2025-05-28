import imaplib
import email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import re
from dotenv import load_dotenv
import requests
from time import sleep
from datetime import datetime, timedelta
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # ✅ Correct import
#from langchain_groq import ChatGroq
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")

 
# Load environment variables
load_dotenv()
 
gmail_address = os.getenv("GMAIL_ADDRESS")
gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")
check_every_n_seconds = int(os.getenv("CHECK_INTERVAL", 30))
 
def send_email(sender, receiver, subject, body):
    print("sending email")
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = f"Re: {subject}"
    msg.attach(MIMEText(body, 'plain'))
   
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender, gmail_app_password)
        server.sendmail(sender, receiver, msg.as_string())
 
    print(f"Sent email to {receiver} with subject: {subject}")
    print("send_email_completed")
 
def fetch_new_emails():
    print("fetch_new_email")
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(gmail_address, gmail_app_password)
    mail.select("INBOX")
   
    since_date = (datetime.now() - timedelta(days=1)).strftime("%d-%b-%Y")
    result, data = mail.search(None, f'(UNSEEN SINCE "{since_date}")')
    new_email_ids = data[0].split()
   
    emails = []
    for email_id in new_email_ids:
        result, data = mail.fetch(email_id, "(RFC822)")
        msg = email.message_from_bytes(data[0][1])
        sender = msg["From"].split("<")[-1].split(">")[0]
        subject = msg["Subject"]
       
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = msg.get_payload(decode=True).decode()
       
        emails.append((email_id, sender, subject, body))
   
    #----------------------NEW CHANGE--------------------------------------
    for email_id in new_email_ids:
        mail.store(email_id, '+FLAGS', '\Seen')
    #----------------------NEW CHANGE--------------------------------------
    print("fetch_new_email_completed")
    return mail, emails
 
def generate_response_with_llama(prompt,email):
    print("Generate _response_with_llama")
    # Combine email details into one string
    email_content = f"{prompt}"
    
    # Define the username
    match = re.match(r"([a-zA-Z._%+-]+)", email)  # Extract only letters and symbols before '@'
    username=match.group(1) if match else None

    # Define the system prompt
    system_prompt = """You are an AI assistant for a shopping mall's customer care support team. Your task is to read incoming customer emails and generate a polite, professional, and empathetic response.
 
        NOTE: Don't say like 'Here is a polite response:'and 'Thank you again' likewise statemennt in the mail. 

        Generate a customer response email that is:
        
        1. Humble and respectful – Sincerely address the customer's concern with a respectful tone.  
        2. Easy to understand – Use simple, professional language without jargon.  
        3. Customer-focused – Acknowledge the issue, briefly restate it to show understanding, provide a clear resolution or next steps, and express gratitude for their inquiry.  
        4. Actionable – If applicable, include specific instructions, offer further assistance, and provide relevant contact information (customercare@smsupermalls.com).  
        
        ### **Response Format:**  
        
        Subject: [Relevant to the customer's issue]  
        
        Dear Customer,  
        
        Thank you for reaching out. We understand your concern regarding [brief restatement of the issue]. [Provide a clear solution or next steps with empathy]. If you need further assistance, feel free to contact us at customercare@smsupermalls.com.        
        Best regards,  
        Shopping Mall Customer Support Team  
    """
 
    # Create a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Email content:\n\n{email_content}\n\nWrite a polite reply."),
        ]
    )
 
    # Instantiate the LLM and chain
    #llm = ChatGroq(model="llama3-70b-8192")
    llm = OllamaLLM(model="llama3.2:3b")  # Use the exact model name
    chain = prompt | llm | StrOutputParser()
 
    # Invoke the chain
    try:
        response = chain.invoke({"email_content": email_content})
    except KeyError as e:
        raise ValueError(f"Error invoking the chain: {e}")
    except Exception as e:
        print("Error:", e)
        return "Sorry, I couldn't generate a response at the moment."
   
    print("Generate _response_with_llama_completed")
    return response
 
 
def process_emails():
    print("process _emails")
    mail, emails = fetch_new_emails()
   
    for email_id, sender, subject, body in emails:
        print(f"Processing email from: {sender}, Subject: {subject}")
       
        ai_response = generate_response_with_llama(body,sender)
        send_email(gmail_address, sender, subject, ai_response)
   
    mail.logout()
    print("process _emails_completed")
 
while True:
    process_emails()
    sleep(check_every_n_seconds)
 
 