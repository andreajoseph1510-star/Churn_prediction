import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def get_retention_advice(customer_data: dict, churn_prob: float, user_question: str = None) -> str:

    if user_question:
        prompt = f"""
        A telecom customer has a {churn_prob:.0%} probability of churning.
        
        Customer details:
        - Contract: {customer_data.get('Contract')}
        - Tenure: {customer_data.get('tenure')} months
        - Monthly Charges: ${customer_data.get('MonthlyCharges')}
        - Internet Service: {customer_data.get('InternetService')}
        - Payment Method: {customer_data.get('PaymentMethod')}
        - Tech Support: {customer_data.get('TechSupport')}
        - Online Security: {customer_data.get('OnlineSecurity')}
        
        User question: {user_question}
        
        Answer the question specifically for this customer. Be concise and actionable.
        """
    else:
        prompt = f"""
        A telecom customer has a {churn_prob:.0%} probability of churning.
        
        Customer details:
        - Contract: {customer_data.get('Contract')}
        - Tenure: {customer_data.get('tenure')} months
        - Monthly Charges: ${customer_data.get('MonthlyCharges')}
        - Internet Service: {customer_data.get('InternetService')}
        - Payment Method: {customer_data.get('PaymentMethod')}
        - Tech Support: {customer_data.get('TechSupport')}
        - Online Security: {customer_data.get('OnlineSecurity')}
        
        Give 3 specific retention strategies for this customer.
        Be concise and actionable. Use bullet points.
        """

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text