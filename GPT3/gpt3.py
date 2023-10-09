import os
import openai
import dotenv
dotenv.load_dotenv()

openai.api_key = os.getenv('API_KEY')

response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Write a tagline for an ice cream shop."
)
print(response.choices[0].text)
