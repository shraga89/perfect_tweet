import pandas as pd
import pscore
import numpy as np
d = {"a":{"t":1,"b":7,"c":4},"b":{"t":2,"b":7,"c":1},"g":{"t":1,"b":7,"c":4}}
a=pd.DataFrame.from_dict(d,orient="index")


prop = pscore.PropensityScore(np.array([1,0,0]),a)
print(prop.compute())