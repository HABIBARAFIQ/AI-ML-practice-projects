from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()
class Student(BaseModel):
    name: str = 'Habiba'
    age: Optional[int] = None
    email: EmailStr
    grade: str = Field(..., description="The grade of the student")
    # ... means required 
    cgpa : float = Field(gt=0,lt=10, description="The CGPA of the student, must be between 0 and 10",default=5)

new_student = Student(
    email='abc@example.com',
    cgpa=8.5,
    grade='A'
)
print(new_student)
json_data = new_student.model_dump_json()
print(json_data)