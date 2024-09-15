from typing import List, Dict

def get_normalize_x(x:List[int])->Dict:
  """
    Get the order index of any list item
  """
  return {v: i for i, v in enumerate(x)}

def reverse_dictionary(dct:Dict)->Dict:
  """
  Reverse key and value in a dictionary
  """
  reverse_dct = {}
  for k, v in dct.items():
    reverse_dct[v] = k
  return reverse_dct
