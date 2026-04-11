from openai import OpenAI

# Connect to LM Studio (local AI)
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

def get_retention_advice(customer_data, churn_prob, user_question=None):

    # 🔀 Dynamic prompt based on question
    if user_question:
        prompt = f"""
        Customer churn probability: {churn_prob:.0%}

        User question: {user_question}

        Answer briefly in one line, then give 4 telecom retention actions as bullet points.
        """
    else:
        prompt = f"""
        Customer churn probability: {churn_prob:.0%}

        Give 4 telecom retention actions as bullet points.
        No explanation.
        """

    # 🔗 Call LM Studio
    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content