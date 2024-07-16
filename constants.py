system_template = """
You have to perform the task of extracting information from data.

Extract the data following the format:
{format_instructions}

Example:
Data: 'Invoice Date: 2023-07-16 Invoice Number: 12345 Total Amount: $1500.00'
Extracted Data: {{ "date": "2023-07-16", "monto": "$1500.00", "facture_number": "12345" }}
"""

prompt_template = """
Data:
{data}
"""