from pydantic import BaseModel, Field

class UserHealthProfile(BaseModel):
    Age: int = Field(default=-1)
    Weight: float = Field(default=-1)
    Average_Sleeping_Hours: float = Field(default=-1)
    Has_Diabetes: int = Field(default=-1)
    Has_High_Blood_Pressure: int = Field(default=-1)
    Daily_Exercise: int = Field(default=-1)