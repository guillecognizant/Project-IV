from pydantic import BaseModel, Field

class Facture(BaseModel):
    """Facture Datamodel"""
    date : str = Field(default = None, description = "Date of the facture") 
    monto : str = Field(default = None, description = "Total money amount of the facture")
    facture_number: str = Field(default = None, description = "Identifier of the facture")
    