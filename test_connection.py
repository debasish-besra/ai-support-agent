from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are a helpful customer support agent."}, {"role": "user", "content": "What is your return policy?"}])

print(response.choices[0].message.content)