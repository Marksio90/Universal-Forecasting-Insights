from src.utils.validators import basic_quality_checks
import pandas as pd

def test_quality():
    df = pd.DataFrame({'a':[1,2,3]})
    res = basic_quality_checks(df)
    assert res['health'].rows == 3
