import json
from typing import Dict, Any

class Config:
    def __init__(self, file_path:str) -> None:
        self.file_path = file_path
    
    def save(self, data:Dict[str, Any]) -> None:
        with open(self.file_path, "w") as f:
            json.dump(data, f, sort_keys=True)
            
    def load(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                
        except FileNotFoundError as e:
            print(f"{e.strerror}: {e.filename}")
            
        else:
            return data